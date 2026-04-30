import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PROJECT_ROOT / "worlds" / "crazyflie_world_assignment.wbt"
DEFAULT_FAILED_FILE = PROJECT_ROOT / "failed_assignment_seeds.txt"
DEFAULT_RESULT_DIR = PROJECT_ROOT / ".assignment_seed_results"
GATE_PROGRESS_RE = re.compile(r"Gate progress:\s*(\[\[.*\]\])")


def find_webots(explicit_path):
    if explicit_path:
        return explicit_path

    from_path = shutil.which("webots") or shutil.which("webots.exe")
    if from_path:
        return from_path

    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        executable = Path(webots_home) / "msys64" / "mingw64" / "bin" / "webots.exe"
        if executable.exists():
            return str(executable)

    common_windows_path = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")
    if common_windows_path.exists():
        return str(common_windows_path)

    return "webots"


def parse_gate_progress(line):
    match = GATE_PROGRESS_RE.search(line)
    if match is None:
        return None
    return ast.literal_eval(match.group(1))


def all_gates_passed(gate_progress):
    return bool(gate_progress) and all(all(lap) for lap in gate_progress)


def append_failed_seed(path, seed, reason, gate_progress):
    progress_text = "" if gate_progress is None else f" gate_progress={gate_progress}"
    with path.open("a", encoding="utf-8") as failed_file:
        failed_file.write(f"{seed} # {reason}{progress_text}\n")


def average_columns(rows):
    if not rows:
        return []
    width = max(len(row) for row in rows)
    averages = []
    for index in range(width):
        values = [row[index] for row in rows if len(row) > index]
        averages.append(sum(values) / len(values))
    return averages


def parse_seed_list(seed_values):
    seeds = []
    for value in seed_values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                seeds.append(int(item))
    return seeds


def run_one_seed(webots, world, seed, timeout, port, result_dir, mode):
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"seed_{seed}_port_{port}.json"
    if result_file.exists():
        result_file.unlink()

    env = os.environ.copy()
    env["AERIAL_ASSIGNMENT_SEED"] = str(seed)
    env["AERIAL_ASSIGNMENT_QUIT_AFTER_RUN"] = "1"
    env["AERIAL_ASSIGNMENT_RESULT_FILE"] = str(result_file)

    command = [
        webots,
        "--batch",
        f"--mode={mode}",
        "--no-rendering",
        "--minimize",
        "--stdout",
        "--stderr",
        f"--port={port}",
        str(world),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            timeout_output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
            for line in timeout_output.splitlines():
                gate_progress = parse_gate_progress(line)
                if gate_progress is not None:
                    return False, "timeout", gate_progress, None, timeout_output
        return False, "timeout", None, None, ""

    if result_file.exists():
        with result_file.open("r", encoding="utf-8") as file:
            result = json.load(file)
        gate_progress = result.get("gate_progress")
        lap_times = result.get("lap_times")
        success = bool(result.get("success"))
        reason = "success" if success else "not all gates passed"
        return success, reason, gate_progress, lap_times, completed.stdout

    gate_progress = None
    for line in completed.stdout.splitlines():
        parsed_progress = parse_gate_progress(line)
        if parsed_progress is not None:
            gate_progress = parsed_progress

    if gate_progress is None:
        return False, f"no gate progress output, exit_code={completed.returncode}", gate_progress, None, completed.stdout
    if not all_gates_passed(gate_progress):
        return False, "not all gates passed", gate_progress, None, completed.stdout
    return True, "success", gate_progress, None, completed.stdout


def run_seed_job(webots, world, seed, timeout, port, result_dir, mode, run_index, total_runs):
    print(f"Starting run {run_index}/{total_runs}, seed {seed}, port {port}", flush=True)
    success, reason, gate_progress, lap_times, output = run_one_seed(webots, world, seed, timeout, port, result_dir, mode)
    return {
        "run_index": run_index,
        "seed": seed,
        "port": port,
        "success": success,
        "reason": reason,
        "gate_progress": gate_progress,
        "lap_times": lap_times,
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the assignment simulation repeatedly with random seeds and save failing seeds."
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of random seeds to test.")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds per simulation run.")
    parser.add_argument("--webots", default=None, help="Path to webots executable if it is not on PATH.")
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD, help="Path to the Webots world file.")
    parser.add_argument("--failed-file", type=Path, default=DEFAULT_FAILED_FILE, help="Where to append failing seeds.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Temporary per-run result files.")
    parser.add_argument("--port", type=int, default=1235, help="First Webots port to use. Parallel runs use following ports.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of Webots simulations to run at the same time.")
    parser.add_argument("--mode", default="realtime", choices=["realtime", "fast", "pause"], help="Webots simulation mode.")
    parser.add_argument("--seeds", nargs="*", help="Specific seed(s) to run, separated by spaces or commas.")
    parser.add_argument("--show-output", action="store_true", help="Print full Webots output after each run.")
    args = parser.parse_args()

    webots = find_webots(args.webots)
    world = args.world.resolve()
    failed_file = args.failed_file.resolve()
    result_dir = args.result_dir.resolve()

    if not world.exists():
        raise FileNotFoundError(f"World file not found: {world}")

    seeds = parse_seed_list(args.seeds)
    if seeds:
        args.runs = len(seeds)
    else:
        random_source = random.SystemRandom()
        seeds = [random_source.randint(0, 2**32 - 1) for _ in range(args.runs)]
    parallel = max(1, args.parallel)
    print(f"Running {args.runs} assignment simulations with {webots}")
    print(f"Webots mode: {args.mode}")
    print(f"Parallel simulations: {parallel}")
    print(f"Using Webots ports {args.port} to {args.port + parallel - 1}")
    print(f"Failed seeds will be saved to {failed_file}")

    failures = 0
    successful_lap_times = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        pending = {}
        next_run_index = 1
        available_ports = list(range(args.port, args.port + parallel))

        while next_run_index <= args.runs and available_ports:
            port = available_ports.pop(0)
            seed = seeds[next_run_index - 1]
            future = executor.submit(
                run_seed_job,
                webots,
                world,
                seed,
                args.timeout,
                port,
                result_dir,
                args.mode,
                next_run_index,
                args.runs,
            )
            pending[future] = port
            next_run_index += 1

        while pending:
            for future in as_completed(pending):
                port = pending.pop(future)
                break

            result = future.result()
            available_ports.append(port)
            seed = result["seed"]
            if args.show_output and result["output"]:
                print(result["output"], end="")
            if result["success"]:
                if result["lap_times"]:
                    successful_lap_times.append(result["lap_times"])
                    print(
                        f"Run {result['run_index']}/{args.runs}, seed {seed} succeeded. "
                        f"Lap times: {result['lap_times']}"
                    )
                else:
                    print(f"Run {result['run_index']}/{args.runs}, seed {seed} succeeded.")
            else:
                failures += 1
                append_failed_seed(failed_file, seed, result["reason"], result["gate_progress"])
                print(
                    f"Run {result['run_index']}/{args.runs}, seed {seed} failed: "
                    f"{result['reason']}. Saved to {failed_file}."
                )

            while next_run_index <= args.runs and available_ports:
                next_port = available_ports.pop(0)
                next_seed = seeds[next_run_index - 1]
                next_future = executor.submit(
                    run_seed_job,
                    webots,
                    world,
                    next_seed,
                    args.timeout,
                    next_port,
                    result_dir,
                    args.mode,
                    next_run_index,
                    args.runs,
                )
                pending[next_future] = next_port
                next_run_index += 1

    print(f"\nDone. Tested {args.runs} seeds, found {failures} failures.")
    if successful_lap_times:
        average_laps = average_columns(successful_lap_times)
        racing_laps = [lap_times[1:] for lap_times in successful_lap_times if len(lap_times) > 1]
        average_racing_laps = average_columns(racing_laps)
        all_racing_times = [time for lap_times in racing_laps for time in lap_times]
        print(f"Successful runs: {len(successful_lap_times)}")
        print(f"Average lap times: {[round(value, 3) for value in average_laps]}")
        if all_racing_times:
            average_racing_time = sum(all_racing_times) / len(all_racing_times)
            print(f"Average racing lap times: {[round(value, 3) for value in average_racing_laps]}")
            print(f"Average racing lap time: {average_racing_time:.3f}")
    else:
        print("Successful runs: 0")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped by user.")
