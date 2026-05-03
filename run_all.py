"""Parallel experiment dispatcher for PA3.

Enumerates every experiment x seed across all four sections and runs them as independent
subprocesses. Each subprocess:
  - cd's into its section folder
  - invokes that section's `run_experiment.py` CLI
  - writes its log to `<section>/logs/<tag>_seed<k>.json`
  - skips silently if the log already exists (resumable)

Isolation: each job is a fresh Python process => fresh CUDA context, independent failures.

Usage:
    # Smoke test: 2 seeds, short budgets (~15 min wall-clock total on RTX 5090)
    python run_all.py --smoke --workers 4

    # Full run: 15 seeds, TA-mandated budgets
    python run_all.py --full --workers 6

    # Subset: only section 2.3 (Reacher) with the full budget
    python run_all.py --full --sections 2.3 --workers 4

    # Resume (just re-run, skips any completed jobs)
    python run_all.py --full --workers 6

Tuning workers:
    * RTX 5090 32GB fits ~6 concurrent SAC agents without VRAM pressure.
    * Reacher (MuJoCo) is more CPU-bound; on a 24-core Ultra 9 you can push 8-10 there.
    * If you see CUDA OOM, drop --workers.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_LOG_DIR = os.path.join(REPO_ROOT, "run_logs")


# =============================================================================
# Experiment catalogue
# =============================================================================
@dataclass
class Job:
    section: str          # "2.1", "2.2", "2.3", "3"
    tag: str              # log filename stem -> <section>/logs/<tag>_seed<seed>.json
    cli_args: list        # args to pass to section's run_experiment.py
    seed: int

    def output_path(self) -> str:
        return os.path.join(REPO_ROOT, self.section, "logs", f"{self.tag}_seed{self.seed}.json")

    def section_dir(self) -> str:
        return os.path.join(REPO_ROOT, self.section)


def build_catalogue(*, smoke: bool, sections: list[str] | None,
                     n_seeds_override: int | None = None) -> list[Job]:
    """Enumerate every (experiment, seed) combo.

    n_seeds_override: if set, use this many seeds per experiment instead of the default
    (2 for smoke, 15 for full). Useful when the full run has to fit a tight deadline.
    """
    jobs: list[Job] = []

    if smoke:
        N_SEEDS = 2
        P = {
            "pend_total":   20_000, "pend_eval":   5_000,
            "ll_cont_total": 40_000, "ll_disc_total": 40_000,
            "ll_eval":       5_000,
            "ll_hover_pre":  30_000, "ll_hover_post": 30_000,
            "reach_total":   40_000, "reach_eval":    5_000,
            "pebble_pend_total":  40_000,
            "pebble_reach_total": 40_000,
            "pebble_budget_pend": 200,
            "pebble_budget_reach": 200,
        }
        ALPHA_GRID = [0.05, 0.1, 0.2, 0.5]
        N_SEEDS_GRID = 1
        BUDGETS = [50, 200]
    else:
        N_SEEDS = n_seeds_override if n_seeds_override is not None else 15
        # Budgets sized with headroom. Pendulum converges well under 100K; LunarLander
        # continuous usually solves at 200-250K so 300K is safe; hover-switch restored
        # to 300K/300K because post-flip "un-learning" is the slowest part of that
        # experiment; PEBBLE-Pendulum restored to 200K because the 5K unsup phase +
        # reward-model learning leaves less effective budget than vanilla SAC.
        P = {
            "pend_total":       100_000, "pend_eval":    10_000,
            "ll_cont_total":    300_000, "ll_disc_total": 250_000,
            "ll_eval":           10_000,
            "ll_hover_pre":     300_000, "ll_hover_post": 300_000,
            "reach_total":      500_000, "reach_eval":    10_000,
            "pebble_pend_total":  200_000,
            "pebble_reach_total": 500_000,
            "pebble_budget_pend":    500,
            "pebble_budget_reach":  1000,
        }
        ALPHA_GRID = [0.05, 0.1, 0.2, 0.5]
        N_SEEDS_GRID = 2
        BUDGETS = [50, 200, 500, 1000]

    want = lambda s: (sections is None) or (s in sections)

    # -------------------- 2.1 Pendulum --------------------
    if want("2.1"):
        TARGET_ANGLES = [0, -10, 30, -60, 90, -90, 120, -150]
        for theta in TARGET_ANGLES:
            for seed in range(N_SEEDS):
                jobs.append(Job("2.1", f"auto_theta{theta}",
                                 ["--kind", "auto", "--theta", str(theta),
                                  "--seed", str(seed),
                                  "--total-steps", str(P["pend_total"]),
                                  "--eval-every", str(P["pend_eval"])], seed))
        MANUAL_ANGLES = [-60, 90, 120, -150]
        # Alpha-grid search: N_SEEDS_GRID seeds per (angle, alpha). After this runs,
        # open 2.1/2_1_pendulum.ipynb, let it pick alpha_mnl per angle from the cache,
        # then the notebook (or a re-run of run_all.py) fills in the remaining seeds
        # only for the winning alpha.
        for theta in MANUAL_ANGLES:
            for alpha in ALPHA_GRID:
                for seed in range(N_SEEDS_GRID):
                    jobs.append(Job("2.1", f"manual_a{alpha}_theta{theta}",
                                     ["--kind", "manual", "--theta", str(theta),
                                      "--seed", str(seed), "--alpha", str(alpha),
                                      "--total-steps", str(P["pend_total"]),
                                      "--eval-every", str(P["pend_eval"])], seed))
        # reward scaling (theta=90, two scales x {auto, manual})
        for s in [10.0, 0.1]:
            for mode in ["manual", "auto"]:
                for seed in range(N_SEEDS):
                    alpha_arg = ["--alpha", str(0.2)] if mode == "auto" else ["--alpha", "0.2"]
                    jobs.append(Job("2.1", f"scale{s}_{mode}_theta90",
                                     ["--kind", "scale", "--theta", "90",
                                      "--seed", str(seed), "--scale", str(s),
                                      "--mode", mode,
                                      "--total-steps", str(P["pend_total"]),
                                      "--eval-every", str(P["pend_eval"])] + alpha_arg,
                                     seed))

    # -------------------- 2.2 LunarLander --------------------
    if want("2.2"):
        for seed in range(N_SEEDS):
            jobs.append(Job("2.2", "cont_auto",
                             ["--kind", "continuous", "--seed", str(seed),
                              "--total-steps", str(P["ll_cont_total"]),
                              "--eval-every", str(P["ll_eval"])], seed))
            for mode in ["fixed", "auto"]:
                jobs.append(Job("2.2", f"hover_{mode}",
                                 ["--kind", "hover", "--seed", str(seed),
                                  "--mode", mode,
                                  "--steps-pre", str(P["ll_hover_pre"]),
                                  "--steps-post", str(P["ll_hover_post"]),
                                  "--eval-every", str(P["ll_eval"])], seed))
            jobs.append(Job("2.2", "disc_sac",
                             ["--kind", "disc", "--seed", str(seed),
                              "--total-steps", str(P["ll_disc_total"]),
                              "--eval-every", str(P["ll_eval"])], seed))
            jobs.append(Job("2.2", "dqn",
                             ["--kind", "dqn", "--seed", str(seed),
                              "--total-steps", str(P["ll_disc_total"]),
                              "--eval-every", str(P["ll_eval"])], seed))

    # -------------------- 2.3 Reacher --------------------
    if want("2.3"):
        for r in ["Ra", "Rb", "Rc"]:
            for seed in range(N_SEEDS):
                jobs.append(Job("2.3", f"reacher_{r}",
                                 ["--reward", r, "--seed", str(seed),
                                  "--total-steps", str(P["reach_total"]),
                                  "--eval-every", str(P["reach_eval"])], seed))

    # -------------------- 3 PEBBLE --------------------
    if want("3"):
        PEND_ANGLES = [0, -60, 90, 120, -150]
        # SAC with GT reward
        for theta in PEND_ANGLES:
            for seed in range(N_SEEDS):
                jobs.append(Job("3", f"sacgt_theta{theta}",
                                 ["--kind", "sac_gt", "--theta", str(theta),
                                  "--seed", str(seed),
                                  "--total-steps", str(P["pebble_pend_total"]),
                                  "--eval-every", str(P["pend_eval"])], seed))
        # PEBBLE on Pendulum (default budget)
        for theta in PEND_ANGLES:
            for seed in range(N_SEEDS):
                jobs.append(Job("3", f"pebble_pend_theta{theta}",
                                 ["--kind", "pebble_pend", "--theta", str(theta),
                                  "--seed", str(seed),
                                  "--budget", str(P["pebble_budget_pend"]),
                                  "--total-steps", str(P["pebble_pend_total"]),
                                  "--eval-every", str(P["pend_eval"])], seed))
        # PEBBLE budget ablation at theta=90 (extras — the default-budget run above handles that one)
        for b in BUDGETS:
            if b == P["pebble_budget_pend"]:
                continue  # already covered above with tag pebble_pend_theta90
            for seed in range(N_SEEDS):
                tag = f"pebble_budget{b}_theta90"
                jobs.append(Job("3", tag,
                                 ["--kind", "pebble_pend", "--theta", "90",
                                  "--seed", str(seed),
                                  "--budget", str(b),
                                  "--tag", tag,
                                  "--total-steps", str(P["pebble_pend_total"]),
                                  "--eval-every", str(P["pend_eval"])], seed))
        # PEBBLE on Reacher, 3 teachers
        for r in ["Ra", "Rb", "Rc"]:
            for seed in range(N_SEEDS):
                jobs.append(Job("3", f"pebble_reacher_{r}",
                                 ["--kind", "pebble_reacher", "--reward", r,
                                  "--seed", str(seed),
                                  "--budget", str(P["pebble_budget_reach"]),
                                  "--total-steps", str(P["pebble_reach_total"]),
                                  "--eval-every", str(P["reach_eval"])], seed))

    # de-dup in case the catalogue produced overlapping jobs
    seen = set(); deduped = []
    for j in jobs:
        key = (j.section, j.tag, j.seed, tuple(j.cli_args))
        if key in seen: continue
        seen.add(key); deduped.append(j)

    # Order longest-running first: 2.3 Reacher and 3 PEBBLE are heaviest.
    # Parse --total-steps out of cli_args for a rough cost proxy.
    def _cost(job):
        steps = 0
        a = job.cli_args
        for i, v in enumerate(a):
            if v == "--total-steps" and i + 1 < len(a):
                steps = max(steps, int(a[i + 1]))
            if v == "--steps-pre" and i + 1 < len(a):
                steps += int(a[i + 1])
            if v == "--steps-post" and i + 1 < len(a):
                steps += int(a[i + 1])
        # heavier cost for Reacher (MuJoCo) and PEBBLE (reward-model training + relabelling)
        weight = 1.0
        if job.section == "2.3": weight *= 1.6
        if job.section == "3" and "pebble" in job.tag: weight *= 1.3
        return steps * weight
    deduped.sort(key=_cost, reverse=True)
    return deduped


# =============================================================================
# Host probe — what can this machine actually handle?
# =============================================================================
def _probe_host():
    import shutil
    n_cpu = os.cpu_count() or 1
    print("=" * 60)
    print("HOST CAPACITY PROBE")
    print("=" * 60)
    print(f"CPU cores (os.cpu_count): {n_cpu}")

    # system RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1]); print(f"system RAM: {kb/1024/1024:.1f} GB"); break
    except FileNotFoundError:
        # macOS fallback
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            print(f"system RAM: {int(out)/1024**3:.1f} GB")
        except Exception:
            print("system RAM: unknown")

    # GPU
    gpu_mem_gb = None
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"], text=True).strip()
            print("GPUs:")
            for line in q.splitlines():
                name, mem_t, mem_f = [x.strip() for x in line.split(",")]
                print(f"  {name}  total={mem_t}  free={mem_f}")
                # crude parse of the first GPU total for sizing
                if gpu_mem_gb is None:
                    gpu_mem_gb = float(mem_t.split()[0]) / 1024  # MiB -> GiB
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
    else:
        print("no nvidia-smi (CPU-only host)")

    # Safe worker recommendation.
    # Rule-of-thumb: one SAC/PEBBLE worker uses ~1.2 GB VRAM + ~800 MB RAM and
    # can usefully use ~2 CPU threads. Reacher (MuJoCo) is a bit more CPU-heavy.
    if gpu_mem_gb:
        vram_workers = max(1, int(gpu_mem_gb * 0.85 / 1.5))  # 1.5 GB headroom per worker
    else:
        vram_workers = 1
    cpu_workers = max(1, n_cpu // 2)
    rec = min(vram_workers, cpu_workers)
    print()
    print(f"Recommendation: --workers {rec}  "
          f"(vram-bound: {vram_workers}, cpu-bound: {cpu_workers}; take the min)")
    print(f"With --workers {rec}, each worker gets {max(1, n_cpu // rec)} CPU threads.")
    print()
    print("Heuristics for pushing higher:")
    print("  * Stable at recommended?  Try workers+2, watch nvidia-smi + htop.")
    print("  * Reacher-heavy workloads tolerate more CPU workers (MuJoCo is CPU-bound).")
    print("  * LunarLander/Pendulum are GPU-light; you can push more there.")
    print("  * If you see CUDA OOM in run_logs/, drop --workers by 2 and retry.")


# =============================================================================
# Dispatcher
# =============================================================================
def _child_env(threads_per_worker: int) -> dict:
    """Environment for a subprocess worker.

    CRITICAL: pin thread pools to `threads_per_worker`. Without this, PyTorch/MKL/OMP
    each grab `nproc` threads per worker, so running N workers in parallel spawns
    N x 24 competing threads on a 24-core box and the whole thing thrashes.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_worker)
    env["MKL_NUM_THREADS"] = str(threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    env["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads_per_worker)
    # PyTorch respects these two via env even before torch is imported
    env.setdefault("TORCH_NUM_THREADS", str(threads_per_worker))
    return env


def _run_job(job_dict: dict) -> dict:
    """Subprocess worker. Runs one experiment end-to-end."""
    job = Job(**{k: job_dict[k] for k in ("section", "tag", "cli_args", "seed")})
    threads_per_worker = int(job_dict.get("threads_per_worker", 1))
    out_path = job.output_path()
    log_name = f"{job.section}__{job.tag}__seed{job.seed}.log"
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    log_path = os.path.join(RUN_LOG_DIR, log_name)

    if os.path.exists(out_path):
        return {"status": "skipped", "job": job_dict, "log": log_path}

    cmd = [sys.executable, "run_experiment.py"] + job.cli_args
    t0 = time.time()
    try:
        with open(log_path, "w") as f:
            f.write(f"# cmd: {shlex.join(cmd)}\n# cwd: {job.section_dir()}\n")
            f.write(f"# threads_per_worker: {threads_per_worker}\n\n")
            f.flush()
            p = subprocess.Popen(cmd, cwd=job.section_dir(),
                                 stdout=f, stderr=subprocess.STDOUT,
                                 env=_child_env(threads_per_worker))
            rc = p.wait()
        dur = time.time() - t0
        return {"status": "ok" if rc == 0 else "fail",
                "rc": rc, "duration_s": dur, "job": job_dict, "log": log_path}
    except Exception as e:
        return {"status": "error", "error": str(e), "job": job_dict, "log": log_path}


def main():
    # Short-circuit --probe so users don't have to pass --smoke/--full.
    if "--probe" in sys.argv[1:]:
        _probe_host(); return

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="2 seeds, reduced budgets")
    g.add_argument("--full",  action="store_true", help="15 seeds, TA-mandated budgets")
    ap.add_argument("--workers", type=int, default=4,
                    help="number of concurrent subprocesses (default 4)")
    ap.add_argument("--threads-per-worker", type=int, default=None,
                    help="CPU threads per worker for OMP/MKL. "
                         "Default = max(1, floor(os.cpu_count()/workers)). "
                         "Pin this to avoid thread-oversubscription.")
    ap.add_argument("--sections", nargs="+", default=None,
                    choices=["2.1", "2.2", "2.3", "3"],
                    help="restrict to these section subset")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate jobs but don't launch")
    ap.add_argument("--probe", action="store_true",
                    help="print host capacity (CPU/RAM/GPU) and a worker-count "
                         "recommendation, then exit")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the seed count (default 2 for --smoke, 15 for --full). "
                         "Use --seeds 5 to trim the full run to a deadline.")
    args = ap.parse_args()

    if args.probe:
        _probe_host()
        return

    n_cpu = os.cpu_count() or 1
    tpw = args.threads_per_worker or max(1, n_cpu // max(1, args.workers))

    jobs = build_catalogue(smoke=args.smoke, sections=args.sections,
                            n_seeds_override=args.seeds)
    total = len(jobs)
    todo = [j for j in jobs if not os.path.exists(j.output_path())]
    done = total - len(todo)

    print(f"catalogue: {total} jobs total, {done} already done, {len(todo)} to run")
    print(f"host cpu : {n_cpu} cores")
    print(f"workers  : {args.workers}  (threads per worker: {tpw}; total threads: "
          f"{args.workers * tpw})")
    if args.dry_run:
        for j in todo[:25]:
            print(f"  [{j.section}] {j.tag} seed={j.seed}")
        if len(todo) > 25:
            print(f"  ... ({len(todo)-25} more)")
        return

    if not todo:
        print("all jobs already complete.")
        return

    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    t0 = time.time()
    completed = 0; failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_job, {**j.__dict__, "threads_per_worker": tpw}): j
                   for j in todo}
        for fut in as_completed(futures):
            res = fut.result()
            j = futures[fut]
            if res["status"] == "ok":
                completed += 1
                print(f"[{completed}/{len(todo)}] OK  [{j.section}] {j.tag} seed={j.seed}  "
                      f"({res['duration_s']:.0f}s)  log={res['log']}")
            elif res["status"] == "skipped":
                completed += 1
                print(f"[{completed}/{len(todo)}] SKP [{j.section}] {j.tag} seed={j.seed}")
            else:
                failed += 1
                print(f"[--]  FAIL [{j.section}] {j.tag} seed={j.seed}  "
                      f"status={res['status']} rc={res.get('rc')} see {res['log']}")
    dur = time.time() - t0
    print(f"\nFinished {completed}/{len(todo)} in {dur/60:.1f} min ({failed} failed)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
