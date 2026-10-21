import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def create_charts(scenario, results_dir="tmp/results"):
    csv_file = Path(results_dir) / scenario / "summary.csv"
    if not csv_file.exists():
        print(f"No summary.csv found for {scenario}")
        return

    df = pd.read_csv(csv_file)
    if scenario == "rover":
        guidance = df["guidance"].fillna("model-only") if "guidance" in df.columns else "model-only"
        df["model"] = df["model"].astype(str) + " [" + guidance + "]"
    
    # Calculate success (0 collisions)
    if "collisions" in df.columns:
        df["success"] = (df["collisions"] == 0).astype(int)
    
    out_dir = Path(results_dir) / scenario / "charts"
    out_dir.mkdir(exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Success Rate
    if "success" in df.columns:
        plt.figure(figsize=(10, 6))
        success_rate = df.groupby("model")["success"].mean() * 100
        sns.barplot(x=success_rate.index, y=success_rate.values, palette="viridis")
        plt.title(f"{scenario.title()} - Success Rate (%)")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "success_rate.png")
        plt.close()
        
    # 2. Distance Traveled
    dist_col = "distance_travelled" if scenario == "rover" else "distance_flown_m"
    if dist_col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="model", y=dist_col, data=df, palette="viridis")
        plt.title(f"{scenario.title()} - Distance Traveled (m)")
        plt.ylabel("Distance (m)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "distance.png")
        plt.close()
        
    # 3. API Calls vs Success
    if "api_calls" in df.columns and "success" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="api_calls", y=dist_col, hue="model", style="success", data=df, s=100)
        plt.title(f"{scenario.title()} - Distance vs API Calls")
        plt.xlabel("API Calls")
        plt.ylabel("Distance (m)")
        plt.tight_layout()
        plt.savefig(out_dir / "api_calls_vs_distance.png")
        plt.close()

    # 4. Total Tokens Cost
    if "total_tokens" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="model", y="total_tokens", data=df, errorbar='sd', palette="magma")
        plt.title(f"{scenario.title()} - Total Tokens per Run")
        plt.ylabel("Tokens")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "tokens.png")
        plt.close()
        
    print(f"Charts saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["drone", "rover", "tunnel", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="tmp/results")
    args = parser.parse_args()
    
    if args.scenario == "all":
        create_charts("drone", args.results_dir)
        create_charts("rover", args.results_dir)
        create_charts("tunnel", args.results_dir)
    else:
        create_charts(args.scenario, args.results_dir)
