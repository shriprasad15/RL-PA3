"""Run a Pendulum SAC job manifest with multiprocessing and live progress monitoring.

This version shows BOTH:
1. job-level progress: completed seed jobs / total jobs
2. step-level progress: aggregate training timesteps from per-seed status.json files

Usage:
    python run_pendulum_jobs.py --manifest outputs/.../auto_jobs.jsonl --workers 4 --poll-interval 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from pendulum_sac_lib import condition_name, train_one_run


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def job_seed_dir(job: Dict[str, Any]) -> Path:
    cond = condition_name(
        str(job["mode"]),
        float(job["target_angle_deg"]),
        float(job.get("reward_scale", 1.0)),
        job.get("manual_alpha", None),
    )
    return Path(job["output_root"]) / cond / f"seed_{int(job['seed']):02d}"


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def read_step_from_eval_log(path: Path) -> int:
    """Fallback if status.json is absent: read last timestep from eval_log.csv cheaply."""
    if not path.exists():
        return 0
    try:
        # Avoid importing pandas in the runner. Read last non-empty line.
        lines = path.read_text().strip().splitlines()
        if len(lines) <= 1:
            return 0
        header = lines[0].split(",")
        values = lines[-1].split(",")
        idx = header.index("timestep")
        return int(float(values[idx]))
    except Exception:
        return 0


def get_job_status(job: Dict[str, Any]) -> Dict[str, Any]:
    seed_dir = job_seed_dir(job)
    total = int(job.get("total_timesteps") or job["base_cfg"]["experiment"]["total_timesteps"])
    done = (seed_dir / "DONE.txt").exists() and (seed_dir / "final_model.pt").exists()
    status = safe_read_json(seed_dir / "status.json") or {}
    if status:
        step = int(status.get("current_step", 0))
        phase = str(status.get("phase", "unknown"))
    else:
        step = read_step_from_eval_log(seed_dir / "eval_log.csv")
        phase = "waiting_or_not_started" if step == 0 else "training_or_evaluating"
    if done:
        step = total
        phase = "completed"
    step = max(0, min(step, total))
    return {
        "seed_dir": str(seed_dir),
        "condition": condition_name(
            str(job["mode"]),
            float(job["target_angle_deg"]),
            float(job.get("reward_scale", 1.0)),
            job.get("manual_alpha", None),
        ),
        "seed": int(job["seed"]),
        "target_angle_deg": float(job["target_angle_deg"]),
        "mode": str(job["mode"]),
        "manual_alpha": job.get("manual_alpha", None),
        "reward_scale": float(job.get("reward_scale", 1.0)),
        "current_step": step,
        "total_timesteps": total,
        "phase": phase,
        "updated_at_unix": float(status.get("updated_at_unix", 0.0)),
        "elapsed_seconds": float(status.get("elapsed_seconds", 0.0)),
        "alpha": status.get("alpha", None),
        "last_eval_return": status.get("last_eval_return", None),
        "best_eval_return": status.get("best_eval_return", None),
        "replay_size": status.get("replay_size", None),
        "pid": status.get("pid", None),
    }


def summarize_active(statuses: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    active = [s for s in statuses if s["phase"] not in {"completed", "skipped_done"} and s["current_step"] > 0]
    active.sort(key=lambda s: (s.get("updated_at_unix", 0.0), s.get("current_step", 0)), reverse=True)
    lines = []
    for s in active[:limit]:
        total = max(1, int(s["total_timesteps"]))
        pct = 100.0 * int(s["current_step"]) / total
        alpha = s.get("alpha")
        ret = s.get("last_eval_return")
        alpha_txt = "NA" if alpha is None else f"{float(alpha):.4g}"
        ret_txt = "NA" if ret is None else f"{float(ret):.1f}"
        lines.append(
            f"theta={s['target_angle_deg']:>6.1f} seed={s['seed']:02d} "
            f"step={int(s['current_step']):>6}/{total:<6} ({pct:5.1f}%) "
            f"phase={s['phase']:<12} alpha={alpha_txt:<8} last_eval={ret_txt:<9} "
            f"pid={s.get('pid', 'NA')}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL file containing jobs")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    parser.add_argument("--results", default=None, help="Optional JSONL result path")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between live progress refreshes")
    parser.add_argument("--status-every-steps", type=int, default=2000, help="Each worker writes status.json every N train steps")
    parser.add_argument("--torch-num-threads", type=int, default=1, help="PyTorch CPU threads per worker. Keep this at 1 for multiprocessing.")
    parser.add_argument("--print-active", type=int, default=10, help="Number of active seed runs to print on each refresh")
    parser.add_argument("--no-active-print", action="store_true", help="Only show progress bar, do not print active seed table")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    jobs = read_jsonl(manifest)
    results_path = Path(args.results).resolve() if args.results else manifest.with_name(manifest.stem + "_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if not jobs:
        print(f"No jobs found in {manifest}")
        return 0

    for j in jobs:
        # Child tqdm output is messy in multiprocessing. Use status.json files instead.
        j["disable_tqdm"] = True
        j["status_every_steps"] = int(args.status_every_steps)
        j["torch_num_threads"] = int(args.torch_num_threads)

    method = args.start_method
    if method == "fork" and sys.platform.startswith("win"):
        method = "spawn"
    ctx = mp.get_context(method)
    workers = max(1, min(int(args.workers), len(jobs)))

    total_train_steps = sum(int(j.get("total_timesteps") or j["base_cfg"]["experiment"]["total_timesteps"]) for j in jobs)

    print(f"Running {len(jobs)} jobs with {workers} worker(s), start_method={method}")
    print(f"Manifest: {manifest}")
    print(f"Results:  {results_path}")
    print(f"Live status: status.json every {int(args.status_every_steps)} train steps per seed")
    print(f"PyTorch threads per worker: {int(args.torch_num_threads)}")
    print("Tip: if the machine lags, stop and rerun with --workers 4 or --workers 6. Completed DONE.txt runs will be skipped.")

    t0 = time.time()
    completed = 0
    failed = 0
    last_progress_steps = 0
    last_active_print = 0.0

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        future_to_job = {ex.submit(train_one_run, job): job for job in jobs}
        pending = set(future_to_job.keys())

        with tqdm(total=total_train_steps, desc="aggregate train steps", unit="step") as step_bar:
            while pending:
                done, pending = wait(pending, timeout=float(args.poll_interval), return_when=FIRST_COMPLETED)

                for fut in done:
                    try:
                        result = fut.result()
                        result["ok"] = True
                        completed += 1
                    except Exception as e:
                        job = future_to_job.get(fut, {})
                        result = {"ok": False, "error": repr(e), "job": job}
                        failed += 1
                    result["elapsed_total_seconds"] = time.time() - t0
                    append_jsonl(results_path, result)
                    if not result.get("ok", False):
                        tqdm.write("FAILED: " + repr(result))

                statuses = [get_job_status(job) for job in jobs]
                current_steps = sum(int(s["current_step"]) for s in statuses)
                current_steps = max(last_progress_steps, min(current_steps, total_train_steps))
                step_bar.update(current_steps - last_progress_steps)
                last_progress_steps = current_steps

                active_count = sum(1 for s in statuses if s["phase"] not in {"completed", "skipped_done"} and s["current_step"] > 0)
                done_count = sum(1 for s in statuses if s["phase"] in {"completed", "skipped_done"})
                step_bar.set_postfix(
                    jobs=f"{completed}/{len(jobs)}",
                    done_files=f"{done_count}/{len(jobs)}",
                    active=active_count,
                    failed=failed,
                )

                now = time.time()
                if (not args.no_active_print) and now - last_active_print >= max(float(args.poll_interval), 15.0):
                    lines = summarize_active(statuses, limit=int(args.print_active))
                    if lines:
                        tqdm.write("\nActive/recent seed status:")
                        for line in lines:
                            tqdm.write("  " + line)
                    else:
                        queued = len(jobs) - done_count
                        tqdm.write(f"\nNo status yet. Jobs may still be importing/initializing/queued. queued_or_running={queued}")
                    last_active_print = now

        # Final accounting check: futures are done at this point.
        for fut in list(pending):
            # Should not normally run because pending is empty.
            pass

    print(f"Done. completed={completed}, failed={failed}, wallclock={time.time() - t0:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
