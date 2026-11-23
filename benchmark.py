import os
import subprocess
import json
import time
import argparse
from pathlib import Path

def _guidance_of(entry, scenario):
    """Rover entries recorded before --guidance existed were all legacy model-only runs."""
    if scenario != "rover":
        return None
    return entry.get("guidance", "model-only")


def _vision_of(entry):
    """Entries recorded before --vision existed were text-only (vision=False)."""
    return bool(entry.get("vision", False))


def run_benchmark(scenario, models, runs, results_dir, timeout_s, record_video=False, provider="auto",
                  guidance="layered", start_run=1, vision=False, seconds=None, skip_existing=False):
    base_dir = Path(results_dir) / scenario
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if scenario == "rover":
        script_name = "run_rover.py"
    elif scenario == "tunnel":
        script_name = "tunnel.py"
    else:
        script_name = "run.py"
    run_guidance = guidance if scenario == "rover" else None
    
    all_results = []
    summary_path = base_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    all_results = existing
        except Exception:
            all_results = []
    if scenario == "rover":
        for x in all_results:
            x.setdefault("guidance", _guidance_of(x, scenario))
    
    for model in models:
        is_heuristic_only = model.lower() in ("heuristic", "baseline", "no-jev")
        for r in range(start_run, runs + 1):
            if is_heuristic_only:
                sanitized_model = "heuristic"
                effective_guidance = "heuristic-only" if scenario == "rover" else None
                effective_vision = False
            else:
                sanitized_model = model.replace("/", "_").replace(":", "_").replace("~", "")
                effective_guidance = run_guidance
                effective_vision = bool(vision)
                # Legacy (model-only) rover runs keep the original "<model>/run_N" layout;
                # layered runs live next to them so the two modes never overwrite each other.
                if effective_guidance and effective_guidance != "model-only":
                    sanitized_model += f"__{effective_guidance}"
                if effective_vision:
                    sanitized_model += "__vision"
            run_dir = base_dir / sanitized_model / f"run_{r}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            log_path = run_dir / "events.jsonl"
            vid_path = run_dir / "video.mp4"
            metrics_path = run_dir / "metrics.json"
            if skip_existing and metrics_path.exists():
                try:
                    with open(metrics_path, "r") as mf:
                        mdata = json.load(mf)
                    vid_ok = (not record_video) or (vid_path.exists() and vid_path.stat().st_size > 1000)
                    if isinstance(mdata, dict) and "error" not in mdata and vid_ok:
                        print(f"Skipping completed {scenario} | Model: {model} | Vision: {effective_vision} | Run {r}/{runs}", flush=True)
                        continue
                except Exception:
                    pass

            if log_path.exists():
                log_path.unlink()
            
            # For headless running on linux/windows
            env = os.environ.copy()
            if os.name != "nt":
                env["MUJOCO_GL"] = "egl"
            if not is_heuristic_only:
                env["JEV_MODEL"] = model
            env["TRACE"] = "1"
            
            cmd = [
                ".venv\\Scripts\\python" if os.name == "nt" else "./.venv/bin/python",
                script_name,
            ]
            if seconds is not None:
                cmd.extend(["--seconds", str(seconds)])
            if is_heuristic_only:
                cmd.append("--no-jev")
            else:
                cmd.extend(["--model", model, "--provider", provider])
            
            if scenario == "rover":
                cmd.extend(["--log", str(log_path), "--headless"])
                if not is_heuristic_only:
                    cmd.extend(["--guidance", effective_guidance])
            elif scenario == "drone":
                cmd.extend(["--export", str(run_dir / "drone_events.json")])
            if effective_vision:
                cmd.append("--vision")
                
            if record_video:
                cmd.extend(["--video", str(vid_path)])
            
            tag = f" | Guidance: {effective_guidance}" if effective_guidance else ""
            if effective_vision:
                tag += " | Vision: ON"
            print(f"Running {scenario} | Model: {model}{tag} | Run {r}/{runs}", flush=True)
            
            start_t = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s
                )
                stdout = proc.stdout
                
                summary = None
                for line in reversed(stdout.strip().split('\n')):
                    try:
                        summary = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                        
                if not summary:
                    print(f"  Warning: No JSON summary found. Exited with code {proc.returncode}", flush=True)
                    summary = {"model": model, "run": r, "error": "No summary", "exit_code": proc.returncode}
                else:
                    if isinstance(summary.get("jev"), dict):
                        jev_stats = summary["jev"]
                        summary.setdefault("api_calls", jev_stats.get("calls", 0))
                        summary.setdefault("images_sent", jev_stats.get("images_sent", 0))
                        summary.setdefault("total_tokens", jev_stats.get("tokens", 0))
                        summary.setdefault("median_latency_s", jev_stats.get("median_latency_s"))
                        summary.setdefault("p90_latency_s", jev_stats.get("p90_latency_s"))
                        summary.setdefault("api_errors", jev_stats.get("errors", 0))
                    summary["run"] = r
                    summary["real_time_s"] = round(time.time() - start_t, 2)
                    print(f"  Done in {summary['real_time_s']:.1f}s | Collisions: {summary.get('collisions', 'N/A')} | API Calls: {summary.get('api_calls', 'N/A')} | Dist: {summary.get('distance_travelled', summary.get('distance_flown_m', 'N/A'))}", flush=True)
                    
                with open(run_dir / "stdout.log", "w") as f:
                    f.write(proc.stdout)
                with open(run_dir / "stderr.log", "w") as f:
                    f.write(proc.stderr)
                    
            except subprocess.TimeoutExpired:
                print(f"  Timed out after {timeout_s}s")
                summary = {"model": model, "run": r, "error": "Timeout"}
            except Exception as e:
                print(f"  Error: {e}")
                summary = {"model": model, "run": r, "error": str(e)}
                
            if is_heuristic_only:
                summary["model"] = "heuristic"
                summary["mode"] = "heuristic"
            else:
                summary["mode"] = "heuristic+llm+image" if effective_vision else "heuristic+llm"
            if effective_guidance:
                summary.setdefault("guidance", effective_guidance)
            summary["vision"] = effective_vision

            with open(run_dir / "metrics.json", "w") as f:
                json.dump(summary, f, indent=2)

            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        latest_on_disk = json.load(f)
                        if isinstance(latest_on_disk, list):
                            all_results = latest_on_disk
                            if scenario == "rover":
                                for x in all_results:
                                    x.setdefault("guidance", _guidance_of(x, scenario))
                except Exception:
                    pass

            target_model_name = "heuristic" if is_heuristic_only else model
            all_results = [
                x for x in all_results
                if not (x.get("model") == target_model_name and x.get("run") == r
                        and _guidance_of(x, scenario) == effective_guidance
                        and _vision_of(x) == effective_vision)
            ]
            all_results.append(summary)
            
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)
            _save_summary_csv(base_dir, all_results)


def _save_summary_csv(base_dir, all_results):
    try:
        import csv
        if all_results:
            flat_results = []
            for r_dict in all_results:
                flat = {}
                for k, v in r_dict.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat[f"{k}_{sub_k}"] = sub_v
                    else:
                        flat[k] = v
                flat_results.append(flat)
                
            keys = set()
            for r_dict in flat_results:
                keys.update(r_dict.keys())
            keys = sorted(list(keys))
            with open(base_dir / "summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flat_results)
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["rover", "tunnel", "drone"], required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--start-run", type=int, default=1,
                        help="Run number to start from (e.g. 2 to add runs 2 and 3 without overwriting run 1)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Simulation duration in seconds (default uses scenario script default)")
    parser.add_argument("--results-dir", type=str, default="tmp/results")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--video", action="store_true", help="Record videos for each run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have valid metrics and video")
    parser.add_argument("--provider", choices=["auto", "typesafe", "native", "openrouter"], default="auto",
                        help="Explicitly force API provider (auto detects from each model name)")
    parser.add_argument("--guidance", choices=["layered", "model-only"], default="layered",
                        help="Rover only: 'layered' (default) keeps baseline_heuristic underneath the model; "
                             "'model-only' is the legacy behaviour (results stored in the original <model>/ dirs)")
    parser.add_argument("--vision", "--send-image", dest="vision", action="store_true",
                        default=os.environ.get("JEV_VISION", "").lower() in ("1", "true", "yes"),
                        help="Render onboard camera RGB frames and send them alongside text telemetry to "
                             "multimodal decision models (results stored in <model>__vision/ dirs).")
    args = parser.parse_args()
    
    run_benchmark(args.scenario, args.models, args.runs, args.results_dir, args.timeout, args.video, args.provider,
                  args.guidance, args.start_run, vision=args.vision, seconds=args.seconds,
                  skip_existing=args.skip_existing)
