import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze(scenario, results_dir="tmp/results"):
    summary_file = Path(results_dir) / scenario / "summary.json"
    if not summary_file.exists():
        print(f"No results found for {scenario} at {summary_file}")
        return

    with open(summary_file, "r") as f:
        results = json.load(f)

    # Group by model (+ guidance mode for rover, + vision flag when enabled)
    models = defaultdict(list)
    for r in results:
        if "error" not in r:
            label = r["model"]
            tags = []
            if scenario == "rover":
                tags.append(r.get("guidance", "model-only"))
            if r.get("vision"):
                tags.append("vision")
            if tags:
                label = f"{label} [{'+'.join(tags)}]"
            models[label].append(r)

    print(f"=== Analysis for {scenario.upper()} ===")
    
    for model, runs in models.items():
        n = len(runs)
        if n == 0:
            continue
            
        successes = sum(1 for r in runs if r.get("collisions", 1) == 0)
        success_rate = (successes / n) * 100
        
        avg_calls = sum(r.get("api_calls", 0) for r in runs) / n
        avg_imgs = sum(r.get("images_sent", 0) for r in runs) / n
        
        dist_key = "distance_travelled" if scenario == "rover" else "distance_flown_m"
        avg_dist = sum(r.get(dist_key, 0) for r in runs) / n
        
        avg_tokens = sum(r.get("total_tokens", 0) for r in runs) / n
        
        avg_time = sum(r.get("real_time_s", 0) for r in runs) / n
        
        print(f"Model: {model}")
        print(f"  Runs: {n}")
        print(f"  Success Rate: {success_rate:.1f}% ({successes}/{n} zero collisions)")
        print(f"  Avg Distance: {avg_dist:.2f}m")
        print(f"  Avg API Calls: {avg_calls:.1f}")
        if avg_imgs > 0:
            print(f"  Avg Images Sent: {avg_imgs:.1f}")
        print(f"  Avg Tokens: {avg_tokens:.1f}")
        print(f"  Avg Real Time: {avg_time:.1f}s")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["drone", "rover", "tunnel", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="tmp/results")
    args = parser.parse_args()
    
    if args.scenario == "all":
        analyze("drone", args.results_dir)
        analyze("rover", args.results_dir)
        analyze("tunnel", args.results_dir)
    else:
        analyze(args.scenario, args.results_dir)
