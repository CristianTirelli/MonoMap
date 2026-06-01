import argparse
import csv
import os
import re
import subprocess
from datetime import datetime

BENCH_DIR = "../benchmarks/b_correct"
TIMEOUT = 4000

# Per-benchmark II table.
II_TABLE = {
    "aes": {2: 16, 5: 16, 10: 16, 20: 16},
    "backprop": {2: 10, 5: 5, 10: 5, 20: 5},
    "basicmath": {2: 7, 5: 7, 10: 7, 20: 7},
    "bitcount": {2: 3, 5: 3, 10: 3, 20: 3},
    "cfd": {2: None, 5: 3, 10: 3, 20: 3},
    "crc32": {2: 11, 5: 11, 10: 11, 20: 11},
    "fft": {2: 7, 5: 7, 10: 7, 20: 7},
    "gsm": {2: 6, 5: 5, 10: 5, 20: 5},
    "heartwall": {2: 9, 5: 3, 10: 3, 20: 3},
    "hotspot3D": {2: 17, 5: 6, 10: 6, 20: 6},
    "lud": {2: 7, 5: 3, 10: 3, 20: 3},
    "nw": {2: 9, 5: 2, 10: 2, 20: 2},
    "particlefilter": {2: 10, 5: 9, 10: 9, 20: 9},
    "sha1": {2: 6, 5: 4, 10: 4, 20: 4},
    "sha2": {2: 7, 5: 7, 10: 7, 20: 7},
    "stringsearch": {2: 7, 5: 3, 10: 3, 20: 3},
    "susan": {2: 6, 5: 2, 10: 2, 20: 2},
}


def get_i_value(bench, x, base_i):
    value = II_TABLE.get(bench, {}).get(x)
    used_fallback = value is None
    if used_fallback:
        return base_i, True
    return value, False


def parse_output(output):
    final_pe_pressure = ""
    recii = ""
    resii = ""
    ii = ""
    nodes = ""
    edges = ""
    max_out_degree = ""
    time_sched_s = ""
    time_space_s = ""
    backtracking_time_space_s = ""
    time_total_s = ""
    monomorphism_found = False
    final_schedule = {}
    reading_schedule = False
    pick_next_stats = ""

    for raw_line in output.splitlines():
        line = raw_line.strip()

        # Detect start of schedule
        if line == "Schedule":
            reading_schedule = True
            continue

        # Read schedule lines
        if reading_schedule:
            # Stop if line doesn't match expected format
            if not re.match(r"^\d+\s+\[.*\]$", line):
                reading_schedule = False
            else:
                parts = line.split(maxsplit=1)
                t = int(parts[0])
                scheduled_nodes = parts[1]
                final_schedule[t] = scheduled_nodes
                continue

        m = re.search(r"max\(([-+]?\d+),\s*([-+]?\d+)\)\s*=\s*([-+]?\d+)", line)
        if m:
            recii = m.group(1)
            resii = m.group(2)

        # Keep the last II printed
        if "Len schedule" in line:
            parts = line.split()
            if parts:
                ii = parts[-1]

        m = re.search(r"#nodes:\s*([^\s]+)", line)
        if m:
            nodes = m.group(1)

        m = re.search(r"#edges:\s*([^\s]+)", line)
        if m:
            edges = m.group(1)

        m = re.search(r"#maxdegree:\s*([^\s]+)", line)
        if m:
            max_out_degree = m.group(1)


        m = re.search(r"End schedule generation:\s*([^\s]+)", line)
        if m:
            time_sched_s = f"{float(m.group(1)):.3f}"

        if "Final PE pressure:" in line:
            final_pe_pressure = line.split("Final PE pressure:")[-1].strip()

        m = re.search(r"Time for monomorphism search:\s*([^\s]+)", line)
        if m:
            time_space_s = f"{float(m.group(1)):.3f}"

        m = re.search(r"Time for backtracking search:\s*([^\s]+)", line)
        if m:
            backtracking_time_space_s = f"{float(m.group(1)):.3f}"

        m = re.search(r"Total time:\s*([^\s]+)", line)
        if m:
            time_total_s = f"{float(m.group(1)):.3f}"

        if "Monomorphism found!" in line:
            monomorphism_found = True

        if "Pick-next stats:" in line:
            pick_next_stats = line.split("Pick-next stats:", 1)[1].strip()

    return {
        "RecII": recii,
        "ResII": resii,
        "II": ii,
        "nodes": nodes,
        "edges": edges,
        "max_out_degree": max_out_degree,
        "time_sched_s": time_sched_s,
        "time_space_s": time_space_s,
        "backtracking_time_space_s": backtracking_time_space_s,
        "time_total_s": time_total_s,
        "monomorphism_found": monomorphism_found,
        "final_pe_pressure": final_pe_pressure,
        "final_schedule": str(final_schedule),
        "pick_next_stats": pick_next_stats,
    }

def ensure_text(value):
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value

def run_benchmark(script, bench, x, y, d, i_value):
    path = f"{BENCH_DIR}/{bench}/kernel1_edges"
    script_path = os.path.abspath(script)

    cmd = [
        "python3",
        script_path,
        "-path",
        path,
        "-x",
        str(x),
        "-y",
        str(y),
        "-d",
        str(d),
        "-i",
        str(i_value),
    ]

    print("RUNNING:", " ".join(cmd))

    output = ""
    stderr = ""
    status = "OK"
    timed_out = False
    exit_code = -1

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        output = ensure_text(proc.stdout)
        stderr = ensure_text(proc.stderr)
        exit_code = proc.returncode

        if proc.returncode != 0:
            status = "ERROR"

    except subprocess.TimeoutExpired as e:
        output = ensure_text(e.stdout)
        stderr = ensure_text(e.stderr)
        status = "TIMEOUT"
        timed_out = True
        exit_code = -1

    parsed = parse_output(output)

    if stderr.strip():
        print("STDERR:")
        print(stderr)

    if not output.strip():
        print("WARNING: empty stdout for benchmark", bench)

    return [
        datetime.now().isoformat(timespec="seconds"),
        bench,
        path,
        status,
        timed_out,
        exit_code,
        x,
        y,
        d,
        i_value,
        parsed["RecII"],
        parsed["ResII"],
        parsed["II"],
        parsed["nodes"],
        parsed["edges"],
        parsed["max_out_degree"],
        parsed["time_sched_s"],
        parsed["time_space_s"],
        parsed["backtracking_time_space_s"],
        parsed["time_total_s"],
        parsed["monomorphism_found"],
        parsed["final_pe_pressure"],
        parsed["final_schedule"],
        parsed["pick_next_stats"],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-script", required=True, help="Python script to run, e.g. monomap.py")
    parser.add_argument("-o", required=True, help="Output CSV file")
    parser.add_argument("-x", type=int, required=True)
    parser.add_argument("-y", type=int, required=True)
    
    parser.add_argument("-d", type=int, required=True)
    parser.add_argument(
        "-base_i",
        type=int,
        default=-1,
        help="Fallback II when II_TABLE has no value for a benchmark/size",
    )

    args = parser.parse_args()

    print("SCRIPT   =", args.script)
    print("BENCH_DIR =", BENCH_DIR)
    print("OUTPUT   =", args.o)
    print("x,y,d    =", args.x, args.y, args.d)
    print("base_i   =", args.base_i)

    if not os.path.isdir(BENCH_DIR):
        raise FileNotFoundError(f"Benchmark directory not found: {BENCH_DIR}")

    if not os.path.isfile(args.script):
        raise FileNotFoundError(f"Script not found: {args.script}")

    benchmarks = sorted(os.listdir(BENCH_DIR))
    print("Benchmarks found:", benchmarks)

    with open(args.o, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_stamp",
            "benchmark",
            "path",
            "status",
            "timed_out",
            "exit_code",
            "x",
            "y",
            "d",
            "i",
            "RecII",
            "ResII",
            "II",
            "nodes",
            "edges",
            "max_out_degree",
            "time_sched_s",
            "time_space_s",
            "backtracking_time_space_s",
            "time_total_s",
            "monomorphism_found",
            "final_pe_pressure",
            "final_schedule",
            "pick_next_stats",
        ])

        for bench in benchmarks:
            bench_path = os.path.join(BENCH_DIR, bench)
            kernel_path = os.path.join(bench_path, "kernel1_edges")

            if not os.path.isdir(bench_path):
                continue

            if not os.path.exists(kernel_path):
                print("Skipping", bench, "- no kernel1_edges")
                continue

            i_value, used_fallback = get_i_value(bench, args.x, args.base_i)
            if used_fallback:
                print(f"{bench} -> using fallback i={i_value}")
            else:
                print(f"{bench} -> using table i={i_value}")

            row = run_benchmark(
                script=args.script,
                bench=bench,
                x=args.x,
                y=args.y,
                d=args.d,
                i_value=i_value,
            )
            writer.writerow(row)

    print("Done. Results written to", args.o)


if __name__ == "__main__":
    main()