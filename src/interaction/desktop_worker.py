"""Isolated engine entry point for the native macOS test app."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import time

MODES = ("voice", "dictation", "gaze", "fusion", "calibrate", "check")


def command_for(mode: str, runtime_dir: Path, execute: bool) -> list[str]:
    commands = {
        "voice": ["voice-live", "--continuous"],
        "dictation": ["voice-dictate", "--continuous"],
        "gaze": ["gaze-live", "--frames", "100000", "--action", "cursor", "--show-dot"],
        "fusion": ["fusion-live"],
        "calibrate": ["gaze-calibrate", "--interactive"],
    }
    if mode not in commands:
        raise ValueError(f"Unsupported desktop mode: {mode}")
    result = [*commands[mode], "--runtime-dir", str(runtime_dir)]
    if mode != "calibrate":
        result.append("--execute" if execute else "--dry-run")
    return result


def failure_messages(payload: object) -> list[str]:
    messages: list[str] = []
    if isinstance(payload, dict):
        if payload.get("status") in {"error", "failure", "blocked"}:
            message = payload.get("message") or payload.get("error")
            if message:
                messages.append(str(message))
        for key, value in payload.items():
            if key not in {"events", "shell_state"}:
                messages.extend(failure_messages(value))
        for event in payload.get("events", []):
            if isinstance(event, dict) and event.get("result"):
                messages.extend(failure_messages(event["result"]))
    elif isinstance(payload, list):
        for value in payload:
            messages.extend(failure_messages(value))
    return list(dict.fromkeys(messages))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--delay", type=float, default=4)
    args = parser.parse_args(argv)
    # All descendants inherit this group, allowing Stop to close camera/audio too.
    if os.getpgrp() != os.getpid():
        os.setsid()
    print("READY\tWorker started", flush=True)
    try:
        from interaction.app import main as run_engine
        for remaining in range(max(0, int(args.delay)), 0, -1):
            print(f"STATUS\tStarting in {remaining}… Click the test pad or switch to your target app.", flush=True)
            time.sleep(1)
        if args.mode == "check":
            print("STATUS\tReady. Choose a mode and press Start.", flush=True)
            return 0
        print("STATUS\tRunning", flush=True)
        output = io.StringIO()
        with redirect_stdout(output):
            run_engine(command_for(args.mode, args.runtime_dir, args.execute))
        payload = json.loads(output.getvalue())
        errors = failure_messages(payload)
        for message in errors:
            print(f"ERROR\t{message}", flush=True)
        if errors:
            return 1
        message = payload.get("final_overlay", {}).get("message")
        print(f"STATUS\t{message or 'Session finished.'}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("STATUS\tStopped", flush=True)
        return 0
    except Exception as error:
        print(f"ERROR\t{type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
