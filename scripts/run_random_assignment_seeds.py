import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PROJECT_ROOT / "worlds" / "crazyflie_world_assignment.wbt"
DEFAULT_FAILED_FILE = PROJECT_ROOT / "failed_assignment_seeds.txt"
DEFAULT_RESULT_DIR = PROJECT_ROOT / ".assignment_seed_results"
GATE_PROGRESS_RE = re.compile(r"Gate progress:\s*(\[\[.*\]\])")
DEBUG_LINE_MARKERS = (
    "Assignment world seed:",
    "Timing started",
    "Lap completed",
    "Gate progress:",
    "Moving to the next segment",
    "Traceback",
    "Exception",
    "[assignment",
)
RACING_DEBUG_MARKERS = (
    "Lap completed",
    "Gate progress:",
    "Moving to the next segment",
    "Traceback",
    "Exception",
    "[assignment",
)


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


def relevant_debug_lines(output, include_detection_lap=False):
    if not output:
        return []

    markers = DEBUG_LINE_MARKERS if include_detection_lap else RACING_DEBUG_MARKERS
    lines = []
    detection_lap_finished = include_detection_lap
    for line in output.splitlines():
        if "Lap completed" in line:
            detection_lap_finished = True
        if detection_lap_finished and any(marker in line for marker in markers):
            lines.append(line)
    return lines


def parse_last_gate_progress(output):
    gate_progress = None
    for line in (output or "").splitlines():
        parsed_progress = parse_gate_progress(line)
        if parsed_progress is not None:
            gate_progress = parsed_progress
    return gate_progress


def write_failure_artifacts(
    result_dir,
    seed,
    port,
    run_index,
    reason,
    command,
    env,
    output,
    gate_progress,
    lap_times,
    result_file,
    timeout,
    mode,
    include_detection_lap=False,
    render=False,
    batch=True,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    failure_dir = result_dir / "failures" / f"seed_{seed}_run_{run_index}_port_{port}_{timestamp}"
    failure_dir.mkdir(parents=True, exist_ok=True)

    output = output or ""
    (failure_dir / "webots.log").write_text(output, encoding="utf-8")
    debug_lines = relevant_debug_lines(output, include_detection_lap=include_detection_lap)
    (failure_dir / "debug_replay.log").write_text("\n".join(debug_lines) + ("\n" if debug_lines else ""), encoding="utf-8")

    result_payload = None
    if result_file.exists():
        try:
            result_payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result_payload = {"unparsed_result_file": result_file.read_text(encoding="utf-8", errors="replace")}
        shutil.copy2(result_file, failure_dir / result_file.name)

    env_snapshot = {
        "AERIAL_ASSIGNMENT_SEED": env.get("AERIAL_ASSIGNMENT_SEED"),
        "AERIAL_ASSIGNMENT_QUIT_AFTER_RUN": env.get("AERIAL_ASSIGNMENT_QUIT_AFTER_RUN"),
        "AERIAL_ASSIGNMENT_RESULT_FILE": env.get("AERIAL_ASSIGNMENT_RESULT_FILE"),
        "PYTHONPATH": env.get("PYTHONPATH"),
        "WEBOTS_HOME": env.get("WEBOTS_HOME"),
    }
    summary = {
        "seed": seed,
        "port": port,
        "run_index": run_index,
        "reason": reason,
        "mode": mode,
        "batch": batch,
        "render": render,
        "timeout": timeout,
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "env": env_snapshot,
        "gate_progress": gate_progress,
        "lap_times": lap_times,
        "result_file": str(result_file),
        "result_payload": result_payload,
        "webots_log": str(failure_dir / "webots.log"),
        "debug_replay_log": str(failure_dir / "debug_replay.log"),
    }
    (failure_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return failure_dir


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


def run_one_seed(
    webots,
    world,
    seed,
    timeout,
    port,
    result_dir,
    mode,
    run_index,
    include_detection_lap=False,
    render=False,
    batch=True,
):
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"seed_{seed}_port_{port}.json"
    if result_file.exists():
        result_file.unlink()

    env = os.environ.copy()
    env["AERIAL_ASSIGNMENT_SEED"] = str(seed)
    env["AERIAL_ASSIGNMENT_QUIT_AFTER_RUN"] = "1"
    env["AERIAL_ASSIGNMENT_RESULT_FILE"] = str(result_file)

    command = [webots]
    if batch:
        command.append("--batch")
    command.extend([f"--mode={mode}", "--stdout", "--stderr", f"--port={port}"])
    if not render:
        command.extend(["--no-rendering", "--minimize"])
    command.append(str(world))

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
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        if result_file.exists():
            with result_file.open("r", encoding="utf-8") as file:
                result = json.load(file)
            gate_progress = result.get("gate_progress")
            lap_times = result.get("lap_times")
            success = bool(result.get("success"))
            reason = "success" if success else "not all gates passed"
            failure_dir = None
            if not success:
                failure_dir = write_failure_artifacts(
                    result_dir,
                    seed,
                    port,
                    run_index,
                    reason,
                    command,
                    env,
                    output,
                    gate_progress,
                    lap_times,
                    result_file,
                    timeout,
                    mode,
                    include_detection_lap=include_detection_lap,
                    render=render,
                    batch=batch,
                )
            return success, reason, gate_progress, lap_times, output, failure_dir

        gate_progress = parse_last_gate_progress(output)
        failure_dir = write_failure_artifacts(
            result_dir,
            seed,
            port,
            run_index,
            "timeout",
            command,
            env,
            output,
            gate_progress,
            None,
            result_file,
            timeout,
            mode,
            include_detection_lap=include_detection_lap,
            render=render,
            batch=batch,
        )
        return False, "timeout", gate_progress, None, output, failure_dir

    if result_file.exists():
        with result_file.open("r", encoding="utf-8") as file:
            result = json.load(file)
        gate_progress = result.get("gate_progress")
        lap_times = result.get("lap_times")
        success = bool(result.get("success"))
        reason = "success" if success else "not all gates passed"
        failure_dir = None
        if not success:
            failure_dir = write_failure_artifacts(
                result_dir,
                seed,
                port,
                run_index,
                reason,
                command,
                env,
                output,
                gate_progress,
                lap_times,
                result_file,
                timeout,
                mode,
                include_detection_lap=include_detection_lap,
                render=render,
                batch=batch,
            )
        return success, reason, gate_progress, lap_times, output, failure_dir

    gate_progress = parse_last_gate_progress(output)

    if gate_progress is None:
        reason = f"no gate progress output, exit_code={completed.returncode}"
        failure_dir = write_failure_artifacts(
            result_dir,
            seed,
            port,
            run_index,
            reason,
            command,
            env,
            output,
            gate_progress,
            None,
            result_file,
            timeout,
            mode,
            include_detection_lap=include_detection_lap,
            render=render,
            batch=batch,
        )
        return False, reason, gate_progress, None, output, failure_dir
    if not all_gates_passed(gate_progress):
        reason = "not all gates passed"
        failure_dir = write_failure_artifacts(
            result_dir,
            seed,
            port,
            run_index,
            reason,
            command,
            env,
            output,
            gate_progress,
            None,
            result_file,
            timeout,
            mode,
            include_detection_lap=include_detection_lap,
            render=render,
            batch=batch,
        )
        return False, reason, gate_progress, None, output, failure_dir
    return True, "success", gate_progress, None, output, None


def run_seed_job(
    webots,
    world,
    seed,
    timeout,
    port,
    result_dir,
    mode,
    run_index,
    total_runs,
    include_detection_lap,
    render,
    batch,
):
    print(f"Starting run {run_index}/{total_runs}, seed {seed}, port {port}", flush=True)
    success, reason, gate_progress, lap_times, output, failure_dir = run_one_seed(
        webots,
        world,
        seed,
        timeout,
        port,
        result_dir,
        mode,
        run_index,
        include_detection_lap=include_detection_lap,
        render=render,
        batch=batch,
    )
    return {
        "run_index": run_index,
        "seed": seed,
        "port": port,
        "success": success,
        "reason": reason,
        "gate_progress": gate_progress,
        "lap_times": lap_times,
        "output": output,
        "failure_dir": failure_dir,
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of Webots simulations to run at the same time. Use 0 to run every requested seed in parallel.",
    )
    parser.add_argument("--mode", default="realtime", choices=["realtime", "fast", "pause"], help="Webots simulation mode.")
    parser.add_argument("--seeds", nargs="*", help="Specific seed(s) to run, separated by spaces or commas.")
    parser.add_argument("--show-output", action="store_true", help="Print full Webots output after each run.")
    parser.add_argument("--render", action="store_true", help="Do not pass --no-rendering/--minimize to Webots.")
    parser.add_argument("--no-batch", action="store_true", help="Do not pass --batch to Webots.")
    parser.add_argument(
        "--include-detection-lap-logs",
        action="store_true",
        help="Include detection-lap lines in failed-run debug_replay.log. Full webots.log is always saved.",
    )
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
    if args.parallel < 0:
        raise ValueError("--parallel must be 0 or greater")
    parallel = args.runs if args.parallel == 0 else min(args.parallel, args.runs)
    print(f"Running {args.runs} assignment simulations with {webots}")
    print(f"Webots mode: {args.mode}")
    print(f"Webots batch: {not args.no_batch}")
    print(f"Webots rendering: {args.render}")
    if not args.render:
        print("GPU rendering: disabled (--no-rendering). Webots physics/controller execution is not forced onto GPU.")
    else:
        print("GPU rendering: enabled where Webots uses it. Physics/controller execution is still CPU-driven.")
    print(f"Parallel simulations: {parallel}")
    print(f"Using Webots ports {args.port} to {args.port + parallel - 1}")
    print(f"Failed seeds will be saved to {failed_file}")
    print(f"Failed-run logs will be saved under {result_dir / 'failures'}")

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
                args.include_detection_lap_logs,
                args.render,
                not args.no_batch,
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
                if result["failure_dir"]:
                    print(f"Failure logs: {result['failure_dir']}")

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
                    args.include_detection_lap_logs,
                    args.render,
                    not args.no_batch,
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
