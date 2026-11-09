"""Manual wrapper for offline synthetic voice fixture generation and benchmarking."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.app import main as interaction_main
from interaction.runtime import (
    VoiceBenchmarkBudget,
    compare_voice_benchmark_summaries,
    evaluate_voice_benchmark_budget,
    render_voice_benchmark_comparison,
    render_voice_benchmark_budget_summary,
    render_voice_benchmark_delta_summary,
    summarize_voice_benchmark_comparison,
)


def _invoke(argv: list[str]) -> dict[str, Any]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = interaction_main(argv)
    output = buffer.getvalue().strip()
    if exit_code != 0:
        raise RuntimeError(f"interaction {' '.join(argv)} failed with exit code {exit_code}")
    if not output:
        return {}
    return json.loads(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic voice fixtures and run the offline voice benchmark. "
            "Run this from a normal terminal session, not a restricted sandbox, "
            "so macOS Speech can transcribe audio files."
        )
    )
    parser.add_argument("--fixtures", default="tests/fixtures/voice/manifests")
    parser.add_argument("--output-root", default="tests/fixtures/voice")
    parser.add_argument("--runtime-dir", default=".interaction")
    parser.add_argument("--active-app", default="Finder")
    parser.add_argument("--manifest-glob", default="*.json")
    parser.add_argument("--interpreter", choices=["regex", "gemma", "hybrid"], default="hybrid")
    parser.add_argument(
        "--compare-interpreters",
        nargs="+",
        choices=["regex", "gemma", "hybrid"],
        default=None,
        help="Run the same corpus against multiple interpreters and print a side-by-side comparison.",
    )
    parser.add_argument(
        "--baseline-interpreter",
        choices=["regex", "gemma", "hybrid"],
        default=None,
        help="Use this interpreter as the baseline run for history, budget, and comparison deltas.",
    )
    parser.add_argument("--budget", default="tests/fixtures/voice/benchmark_budget.json")
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--skip-budget", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "fixtures": str(Path(args.fixtures).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "runtime_dir": str(Path(args.runtime_dir).expanduser().resolve()),
    }
    history_path = (
        Path(str(args.history_path)).expanduser().resolve()
        if args.history_path
        else Path(str(args.runtime_dir)).expanduser().resolve() / "benchmarks" / "voice" / "latest_summary.json"
    )
    payload["benchmark_history_path"] = str(history_path)

    if not args.skip_generate:
        payload["generation"] = _invoke(
            [
                "voice-generate-fixtures",
                "--fixtures",
                str(args.fixtures),
                "--output-root",
                str(args.output_root),
                "--runtime-dir",
                str(args.runtime_dir),
                "--manifest-glob",
                str(args.manifest_glob),
                *(["--force"] if args.force else []),
            ]
        )

    if not args.skip_benchmark:
        previous_summary = None if args.skip_history else _load_previous_summary(history_path)
        interpreters = _resolve_interpreters(args)
        run_payloads = {
            interpreter: _invoke(
                [
                    "voice-benchmark",
                    "--fixtures",
                    str(args.fixtures),
                    "--runtime-dir",
                    str(args.runtime_dir),
                    "--manifest-glob",
                    str(args.manifest_glob),
                    "--active-app",
                    str(args.active_app),
                    "--interpreter",
                    interpreter,
                    *(["--fail-fast"] if args.fail_fast else []),
                ]
            )
            for interpreter in interpreters
        }
        baseline_interpreter = _resolve_baseline_interpreter(
            interpreters,
            preferred=str(args.baseline_interpreter) if args.baseline_interpreter else str(args.interpreter),
        )
        payload["benchmark_interpreter"] = baseline_interpreter
        payload["benchmark_run_order"] = interpreters
        if len(run_payloads) > 1:
            payload["benchmark_runs"] = run_payloads
            comparison = summarize_voice_benchmark_comparison(
                {
                    interpreter: run_payload["summary"]
                    for interpreter, run_payload in run_payloads.items()
                    if isinstance(run_payload, dict) and isinstance(run_payload.get("summary"), dict)
                },
                baseline_label=baseline_interpreter,
            )
            comparison_text = render_voice_benchmark_comparison(comparison)
            payload["benchmark_comparison"] = comparison
            payload["benchmark_comparison_text"] = comparison_text
            print(comparison_text, file=sys.stderr)
        payload["benchmark"] = run_payloads[baseline_interpreter]
        benchmark_summary_text = payload["benchmark"].get("summary_text")
        if isinstance(benchmark_summary_text, str) and benchmark_summary_text.strip():
            payload["benchmark_summary_text"] = benchmark_summary_text
            print(benchmark_summary_text, file=sys.stderr)
        benchmark_delta = compare_voice_benchmark_summaries(previous_summary, payload["benchmark"]["summary"])
        benchmark_delta_text = render_voice_benchmark_delta_summary(benchmark_delta)
        payload["benchmark_delta"] = benchmark_delta
        payload["benchmark_delta_text"] = benchmark_delta_text
        print(benchmark_delta_text, file=sys.stderr)
        if not args.skip_history:
            _write_current_summary(history_path, payload["benchmark"]["summary"])

    exit_code = 0
    if not args.skip_benchmark and not args.skip_budget:
        budget_path = Path(args.budget).expanduser().resolve()
        if not budget_path.is_file():
            raise FileNotFoundError(f"Voice benchmark budget not found: {budget_path}")
        budget = VoiceBenchmarkBudget.load(budget_path)
        budget_result = evaluate_voice_benchmark_budget(payload["benchmark"]["summary"], budget)
        budget_summary_text = render_voice_benchmark_budget_summary(budget_result)
        payload["benchmark_budget"] = budget_result
        payload["benchmark_budget_text"] = budget_summary_text
        print(budget_summary_text, file=sys.stderr)
        if budget_result.get("passed") is not True:
            exit_code = 1

    print(json.dumps(payload, indent=2))
    return exit_code


def _load_previous_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def _resolve_interpreters(args: argparse.Namespace) -> list[str]:
    values = list(args.compare_interpreters or [str(args.interpreter)])
    deduped: list[str] = []
    for value in values:
        interpreter = str(value)
        if interpreter not in deduped:
            deduped.append(interpreter)
    return deduped or [str(args.interpreter)]


def _resolve_baseline_interpreter(interpreters: list[str], *, preferred: str | None) -> str:
    if preferred and preferred in interpreters:
        return preferred
    return interpreters[0]


def _write_current_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().astimezone().isoformat(),
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
