"""Measure workstation hardware/software specs, OpenRouter network latency,
Cloudflare CDN bandwidth, and published dataset baseline determinism.

Reproduces the experimental setup and network overhead measurements reported in
Section 3.3 of the study (TCP connect, TLS handshake, warm HTTPS Keep-Alive RTT,
Cloudflare download/upload throughput, onboard RGB JPEG payload size, and
5-run deterministic baseline parity from the published 410-run dataset).

Usage:
    python scripts/measure_network_specs.py
    python scripts/measure_network_specs.py --dataset dataset_new/som_navigation_benchmark_410.jsonl.gz
    python scripts/measure_network_specs.py --skip-network
"""
import argparse
import gzip
import http.client
import json
import os
import platform
import socket
import ssl
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "dataset_new" / "som_navigation_benchmark_410.jsonl.gz"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def fmt(arr, unit="ms", decimals=2):
    if not arr:
        return "n=0"
    s = sorted(arr)
    n = len(s)
    p90 = s[min(n - 1, int(0.9 * n))]
    return (
        f"n={n}, min={s[0]:.{decimals}f}{unit}, med={statistics.median(s):.{decimals}f}{unit}, "
        f"mean={statistics.mean(s):.{decimals}f}{unit}, sd={statistics.pstdev(s):.{decimals}f}{unit}, "
        f"p90={p90:.{decimals}f}{unit}, max={s[-1]:.{decimals}f}{unit}"
    )


def _pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return "not installed"


def section_1_system_specs():
    print("=== 1. SYSTEM, HARDWARE & SOFTWARE SPECS ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(
        f"MuJoCo: {_pkg_version('mujoco')}, NumPy: {_pkg_version('numpy')}, "
        f"OpenAI SDK: {_pkg_version('openai')}"
    )
    if platform.system() == "Windows":
        try:
            ps_hw = (
                "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress; "
                "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress; "
                "Get-CimInstance Win32_VideoController | Select-Object Name | ConvertTo-Json -Compress"
            )
            hw_out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_hw],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            print("Hardware CIM:", hw_out.strip())
        except Exception as e:
            print("Hardware CIM error:", e)
    else:
        print(f"CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)")


def section_2_network_adapter():
    print("\n=== 2. LOCAL NETWORK ADAPTER INFO ===")
    if platform.system() == "Windows":
        try:
            ps_net = (
                "Get-NetAdapter | Where-Object Status -eq 'Up' | "
                "Select-Object Name,InterfaceDescription,LinkSpeed,MediaType | ConvertTo-Json -Compress"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_net],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            print(out.strip())
        except Exception as e:
            print("NetAdapter error:", e)
    else:
        print("Adapter query via PowerShell is Windows-only; skipped on non-Windows OS.")


def section_3_and_4_openrouter_rtt(host="openrouter.ai", n_probes=20):
    print(f"\n=== 3. TCP & TLS RTT TO {host}:443 ({n_probes} probes) ===")
    tcp_rtts = []
    tls_rtts = []
    ctx = ssl.create_default_context()
    ip = socket.gethostbyname(host)
    print(f"Resolved {host} -> {ip}")

    for i in range(n_probes):
        try:
            t0 = time.perf_counter()
            sock = socket.create_connection((host, 443), timeout=5)
            t1 = time.perf_counter()
            with ctx.wrap_socket(sock, server_hostname=host):
                t2 = time.perf_counter()
            tcp_rtts.append((t1 - t0) * 1000.0)
            tls_rtts.append((t2 - t1) * 1000.0)
        except Exception as e:
            print(f"Probe {i} error: {e}")

    if tcp_rtts:
        print("TCP connect RTT (SYN->SYN-ACK):", fmt(tcp_rtts))
        print("TLS handshake time:            ", fmt(tls_rtts))

    print(
        f"\n=== 4. HTTP KEEP-ALIVE PING TO {host}/api/v1/models "
        f"({n_probes} warm requests on persistent socket) ==="
    )
    conn = http.client.HTTPSConnection(host, 443, timeout=10, context=ctx)
    http_rtts = []
    try:
        for i in range(n_probes + 1):
            t0 = time.perf_counter()
            conn.request("HEAD", "/api/v1/models")
            resp = conn.getresponse()
            resp.read()
            t1 = time.perf_counter()
            if i > 0:  # skip initial cold request
                http_rtts.append((t1 - t0) * 1000.0)
    finally:
        conn.close()

    if http_rtts:
        print("Warm HTTPS HEAD RTT:           ", fmt(http_rtts))

    if tcp_rtts and tls_rtts and http_rtts:
        total_cold_med = (
            statistics.median(tcp_rtts)
            + statistics.median(tls_rtts)
            + statistics.median(http_rtts)
        )
        print(
            f"Estimated per-request cold HTTPS overhead (TCP_med + TLS_med + HEAD_med): "
            f"{total_cold_med:.2f} ms ({total_cold_med / 1000.0:.4f} s)"
        )


def section_5_cloudflare_bandwidth(n_runs=5):
    print(f"\n=== 5. CLOUDFLARE SPEEDTEST / BANDWIDTH PROBE ({n_runs} runs each) ===")
    down_mbps_runs = []
    up_mbps_runs = []
    for _ in range(n_runs):
        req = urllib.request.Request(
            "https://speed.cloudflare.com/__down?bytes=10000000",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        dt = time.perf_counter() - t0
        down_mbps_runs.append((len(data) * 8) / (dt * 1e6))

        payload = b"0" * 5000000
        req_up = urllib.request.Request(
            "https://speed.cloudflare.com/__up",
            data=payload,
            method="POST",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/octet-stream",
            },
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req_up, timeout=15) as resp:
            resp.read()
        dt_up = time.perf_counter() - t0
        up_mbps_runs.append((len(payload) * 8) / (dt_up * 1e6))

    print("Download 10MB (speed.cloudflare.com):", fmt(down_mbps_runs, unit=" Mbps", decimals=1))
    print("Upload   5MB  (speed.cloudflare.com):", fmt(up_mbps_runs, unit=" Mbps", decimals=1))


def _probe_sample_jpeg_payload():
    """Render one 384x288 onboard RGB frame per available MuJoCo scene to report
    live JPEG and Base64 payload sizes without relying on raw local logs."""
    try:
        import mujoco
    except Exception as e:
        print(f"Sample RGB payload check skipped ({e})")
        return

    samples = []
    # Rover scene is self-contained in the repository (no mujoco_menagerie required)
    rover_xml = REPO_ROOT / "rover_world.xml"
    if rover_xml.is_file():
        try:
            from drive import Eye as RoverEye

            m = mujoco.MjModel.from_xml_path(str(rover_xml))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            eye = RoverEye(m, vision=True)
            scene = eye(d, 0.0)
            url = scene["camera_image"]["image_url"]["url"]
            b64_kb = len(url.split(",", 1)[-1]) / 1024.0
            samples.append(("UGV (rover_world.xml)", b64_kb * 0.75, b64_kb))
        except Exception as e:
            print(f"UGV sample frame render skipped ({e})")

    # Drone scene requires mujoco_menagerie/skydio_x2 (fetched by setup.sh)
    drone_xml = REPO_ROOT / "world.xml"
    if drone_xml.is_file():
        try:
            from flight import Eye as DroneEye

            m = mujoco.MjModel.from_xml_path(str(drone_xml))
            d = mujoco.MjData(m)
            d.qpos[:3] = [1.5, 0.0, 1.6]
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(m, d)
            eye = DroneEye(m, vision=True)
            scene = eye.look(d, d.qpos[:3], 0.0, 0.0)
            url = scene["camera_image"]["image_url"]["url"]
            b64_kb = len(url.split(",", 1)[-1]) / 1024.0
            samples.append(("UAV (world.xml)", b64_kb * 0.75, b64_kb))
        except Exception as e:
            print(f"UAV sample frame render skipped ({e})")

    for label, raw_kb, b64_kb in samples:
        print(f"Sample 384x288 JPEG (q=80) [{label}]: raw={raw_kb:.2f} KB, base64={b64_kb:.2f} KB")


def section_6_dataset_and_determinism(dataset_path: Path):
    print(f"\n=== 6. PUBLISHED DATASET SUMMARY, PAYLOAD SIZE & BASELINE DETERMINISM ===")
    if not dataset_path.is_file():
        print(f"Dataset not found at {dataset_path}; skipping dataset verification.")
        _probe_sample_jpeg_payload()
        return

    opener = gzip.open if dataset_path.suffix == ".gz" else open
    drone_h = []
    rover_h = []
    total_runs = 0
    som_runs = 0
    vision_runs = 0
    total_images_sent = 0
    all_t50 = []
    all_t90 = []

    with opener(dataset_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_runs += 1
            dom = rec.get("domain")
            m = rec.get("metrics", {})
            mode = rec.get("mode") or m.get("mode")
            if mode == "heuristic" or rec.get("model") == "heuristic":
                if dom == "drone":
                    drone_h.append((
                        rec.get("run"),
                        m.get("distance_flown_m"),
                        m.get("max_x_m"),
                        m.get("collisions"),
                        m.get("steps_reflex_pct"),
                        m.get("flew_s"),
                        m.get("real_time_s"),
                    ))
                elif dom == "rover":
                    rover_h.append((
                        rec.get("run"),
                        round(float(m.get("distance_travelled", 0.0)), 2),
                        m.get("max_x"),
                        m.get("collisions"),
                        round(float(m.get("final_t", 0.0)), 2),
                        m.get("real_time_s"),
                    ))
            else:
                som_runs += 1
                if rec.get("vision"):
                    vision_runs += 1
                    total_images_sent += int(m.get("images_sent") or 0)
                t50 = m.get("median_latency_s")
                t90 = m.get("p90_latency_s")
                if t50 is not None:
                    all_t50.append(float(t50) * 1000.0)
                if t90 is not None:
                    all_t90.append(float(t90) * 1000.0)

    drone_h.sort(key=lambda x: x[0])
    rover_h.sort(key=lambda x: x[0])

    print(
        f"Dataset: {dataset_path.name} ({total_runs} runs: {som_runs} SOM "
        f"[{vision_runs} multimodal, {total_images_sent} RGB frames sent], "
        f"{len(drone_h) + len(rover_h)} heuristic baselines)"
    )
    if all_t50:
        print("SOM run-level T50 latency:", fmt(all_t50, unit="ms", decimals=1))
    if all_t90:
        print("SOM run-level T90 latency:", fmt(all_t90, unit="ms", decimals=1))

    _probe_sample_jpeg_payload()

    drone_summary = [row[1:] for row in drone_h]
    rover_summary = [row[1:] for row in rover_h]
    drone_det = len({row[:5] for row in drone_summary}) == 1 if drone_summary else False
    rover_det = len({row[:4] for row in rover_summary}) == 1 if rover_summary else False

    print(
        f"Drone heuristic ({len(drone_summary)} runs, deterministic={drone_det}) "
        f"[dist, max_x, col, reflex_pct, sim_s, wall_s]: {drone_summary}"
    )
    print(
        f"Rover heuristic ({len(rover_summary)} runs, deterministic={rover_det}) "
        f"[dist, max_x, col, sim_s, wall_s]: {rover_summary}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Measure system/network specs and verify published dataset baseline determinism."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to published JSONL/JSONL.GZ dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--host", default="openrouter.ai", help="API gateway host for TCP/TLS/HTTPS probes")
    parser.add_argument("--probes", type=int, default=20, help="Number of TCP/TLS and Keep-Alive RTT probes")
    parser.add_argument("--speed-runs", type=int, default=5, help="Number of Cloudflare download/upload runs")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip live network probes (sections 3-5) and only run system + dataset checks",
    )
    args = parser.parse_args()

    section_1_system_specs()
    section_2_network_adapter()
    if not args.skip_network:
        section_3_and_4_openrouter_rtt(host=args.host, n_probes=args.probes)
        section_5_cloudflare_bandwidth(n_runs=args.speed_runs)
    else:
        print("\n=== 3-5. NETWORK PROBES SKIPPED (--skip-network) ===")
    section_6_dataset_and_determinism(args.dataset)


if __name__ == "__main__":
    main()
