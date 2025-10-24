from pathlib import Path
import time
import json
import argparse
from typing import List, Dict, Any, Optional

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from job_shop_lib.constraint_programming import ORToolsSolver
from job_shop_lib.benchmarking import load_benchmark_instance

RESULTS_FILE = Path("results.json")

# -------------------------
# CP-SAT method for LA datasets
# -------------------------
def make_datasets(start: int = 1, end: int = 40) -> List[str]:
    """Generate dataset names la01 .. la40 by default."""
    return [f"la{idx:02d}" for idx in range(start, end + 1)]


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load results saved as a JSON array. If file is not a valid JSON array,
    attempt to parse one JSON object per line."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: parse newline/commas separated JSON objects
        results = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                # ignore malformed lines
                continue
        return results

# -------------------------
# Save CP-SAT results of makespan and time consumption
# -------------------------
def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    """Persist results as a JSON array (overwrites)."""
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(dataset: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Run solver on a single dataset and return a result dict."""
    res: Dict[str, Any] = {
        "dataset": dataset,
        "time_used": None,
        "makespan": None,
        "success": False,
        "error": None,
        "metadata": None,
        "is_complete": None,
    }
    t0 = time.perf_counter()
    try:
        instance = load_benchmark_instance(dataset)
        solver = ORToolsSolver()
        if timeout is not None:
            # If ORToolsSolver supports configuring a timeout via constructor/attributes,
            # user can modify ORToolsSolver accordingly. Here we call it directly.
            schedule = solver(instance)
        else:
            schedule = solver(instance)

        elapsed = time.perf_counter() - t0
        res["time_used"] = round(elapsed, 4)
        try:
            res["makespan"] = schedule.makespan()
        except Exception:
            res["makespan"] = None

        try:
            res["metadata"] = schedule.metadata
        except Exception:
            res["metadata"] = None

        try:
            res["is_complete"] = bool(schedule.is_complete())
        except Exception:
            res["is_complete"] = None

        res["success"] = True
        print(f"{dataset}: OK  makespan={res['makespan']}  time={res['time_used']}s")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        res["time_used"] = round(elapsed, 4)
        res["error"] = str(exc)
        res["success"] = False
        print(f"{dataset}: ERROR after {res['time_used']}s -> {exc}")
    return res

# -------------------------
# Plot makespan and time used vs dataset
# -------------------------
def plot_results(results: List[Dict[str, Any]],
                 out_time: Path = Path("time_used_vs_dataset.png"),
                 out_makespan: Path = Path("makespan_vs_dataset.png")) -> None:
    if not _HAS_MPL:
        print("matplotlib not available: skipping plots")
        return
    if not results:
        print("no results to plot")
        return

    datasets = [r["dataset"] for r in results]
    times = [r.get("time_used", 0.0) for r in results]
    makespans = [r.get("makespan") if r.get("makespan") is not None else float("nan") for r in results]

    plt.figure(figsize=(12, 4))
    plt.plot(datasets, times, marker="o")
    plt.xlabel("Dataset")
    plt.ylabel("Time Used (s)")
    plt.title("Time Used vs Dataset")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_time)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(datasets, makespans, marker="o")
    plt.xlabel("Dataset")
    plt.ylabel("Makespan")
    plt.title("Makespan vs Dataset")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_makespan)
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run OR-Tools solver on benchmark instances.")
    p.add_argument("--start", type=int, default=1, help="Start dataset index (inclusive).")
    p.add_argument("--end", type=int, default=40, help="End dataset index (inclusive).")
    p.add_argument("--results", type=str, default=str(RESULTS_FILE), help="Path to results JSON file.")
    p.add_argument("--no-plot", action="store_true", help="Do not generate plots.")
    args = p.parse_args(argv)

    results_path = Path(args.results)
    results: List[Dict[str, Any]] = load_results(results_path)

    # Determine which datasets still need processing
    all_datasets = make_datasets(args.start, args.end)
    done = {r["dataset"] for r in results}
    pending = [d for d in all_datasets if d not in done]

    for ds in pending:
        r = run_one(ds)
        results.append(r)
        # save incrementally
        save_results(results, results_path)

    # final save (pretty)
    save_results(results, results_path)

    if not args.no_plot:
        plot_results(results)

    return 0


if __name__ == "__main__":
    main()