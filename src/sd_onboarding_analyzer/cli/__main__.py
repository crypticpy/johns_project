from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict


# Imports for CLI helpers are intentionally deferred into main()
# to ensure environment variables (offline defaults, RBAC) are set
# before importing the FastAPI app factory to avoid cached settings.


def _print_error(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.stderr.flush()


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdonb",
        description="Service Desk Onboarding Analyzer – offline-friendly CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser(
        "ingest",
        help="Upload CSV/XLSX and persist dataset+tickets (offline).",
    )
    p_ingest.add_argument(
        "--file",
        required=True,
        help="Path to CSV or Excel file",
    )
    p_ingest.add_argument(
        "--name",
        required=False,
        help="Optional dataset name override",
    )

    # embed
    p_embed = sub.add_parser(
        "embed",
        help="Compute embeddings and build FAISS index for a dataset.",
    )
    p_embed.add_argument(
        "--dataset-id",
        required=True,
        type=int,
        help="Dataset ID",
    )
    p_embed.add_argument(
        "--backend",
        choices=["builtin", "sentence-transformers"],
        default="builtin",
        help="Embedding backend (default: builtin)",
    )
    p_embed.add_argument(
        "--model-name",
        default="builtin-384",
        help="Embedder model name (default: builtin-384)",
    )

    # analysis
    p_analysis = sub.add_parser(
        "analysis",
        help="Run offline analysis for a dataset question.",
    )
    p_analysis.add_argument(
        "--dataset-id",
        required=True,
        type=int,
        help="Dataset ID",
    )
    p_analysis.add_argument(
        "--question",
        required=True,
        help="Analysis question (e.g., 'Top onboarding gaps?')",
    )
    p_analysis.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt template version (default: v1)",
    )
    p_analysis.add_argument(
        "--max-tickets",
        type=int,
        default=50,
        help="Sampling cap (default: 50)",
    )
    p_analysis.add_argument(
        "--token-budget",
        type=int,
        default=8000,
        help="Token budget for context construction (default: 8000)",
    )

    # report
    p_report = sub.add_parser(
        "report",
        help="Retrieve dataset report markdown.",
    )
    p_report.add_argument(
        "--dataset-id",
        required=True,
        type=int,
        help="Dataset ID",
    )
    p_report.add_argument(
        "--out",
        required=False,
        help="Optional output path to write markdown",
    )

    # pipeline
    p_pipeline = sub.add_parser(
        "pipeline",
        help="End-to-end: ingest → embed → analysis → report.",
    )
    p_pipeline.add_argument(
        "--file",
        required=True,
        help="Path to CSV or Excel file",
    )
    p_pipeline.add_argument(
        "--question",
        required=True,
        help="Analysis question (e.g., 'Top onboarding gaps?')",
    )
    p_pipeline.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt template version (default: v1)",
    )
    p_pipeline.add_argument(
        "--backend",
        choices=["builtin", "sentence-transformers"],
        default="builtin",
        help="Embedding backend (default: builtin)",
    )
    p_pipeline.add_argument(
        "--model-name",
        default="builtin-384",
        help="Embedder model name (default: builtin-384)",
    )
    p_pipeline.add_argument(
        "--max-tickets",
        type=int,
        default=50,
        help="Sampling cap (default: 50)",
    )
    p_pipeline.add_argument(
        "--token-budget",
        type=int,
        default=8000,
        help="Token budget for context construction (default: 8000)",
    )
    p_pipeline.add_argument(
        "--out",
        required=False,
        help="Optional output path to write final report markdown",
    )

    return parser


def main() -> None:
    # Enforce offline determinism; prevent network calls
    os.environ.setdefault("APP_EMBED_BACKEND", "builtin")
    os.environ.setdefault("ANALYZER_BACKEND", "offline")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Force RBAC off for CLI flows to avoid token requirements in CI/offline runs
    os.environ["APP_ENABLE_RBAC"] = "false"

    # Defer imports until after env is set to avoid cached settings with incorrect RBAC
    # pylint: disable=import-error, relative-beyond-top-level
    try:
        from sd_onboarding_analyzer.cli.app_factory import get_app  # type: ignore
        from sd_onboarding_analyzer.cli.run_pipeline import (  # type: ignore
            run_ingest,
            run_embed,
            run_analysis,
            run_report,
            run_pipeline,
            to_json,
        )
    except Exception:  # pragma: no cover
        from .app_factory import get_app  # type: ignore
        from .run_pipeline import (  # type: ignore
            run_ingest,
            run_embed,
            run_analysis,
            run_report,
            run_pipeline,
            to_json,
        )

    parser = _build_parser()
    args = parser.parse_args()

    try:
        app = get_app()

        if args.cmd == "ingest":
            payload: Dict[str, Any] = run_ingest(app, args.file, dataset_name=args.name)
            print(to_json(payload))
            return

        if args.cmd == "embed":
            payload = run_embed(
                app,
                args.dataset_id,
                backend=args.backend,
                model_name=args.model_name,
            )
            print(to_json(payload))
            return

        if args.cmd == "analysis":
            payload = run_analysis(
                app,
                args.dataset_id,
                question=args.question,
                prompt_version=args.prompt_version,
                max_tickets=args.max_tickets,
                token_budget=args.token_budget,
            )
            # Print compact summary JSON
            print(to_json(payload))
            return

        if args.cmd == "report":
            payload = run_report(app, args.dataset_id)
            report_md = payload.get("report_markdown", "")
            if args.out:
                out_path = Path(args.out)
                _write_text_file(out_path, report_md)
                print(str(out_path))
            else:
                # Print markdown to stdout
                print(report_md)
            return

        if args.cmd == "pipeline":
            result = run_pipeline(
                app,
                file_path=args.file,
                question=args.question,
                prompt_version=args.prompt_version,
                embed_backend=args.backend,
                embed_model=args.model_name,
                max_tickets=args.max_tickets,
                token_budget=args.token_budget,
            )
            report_md = result.get("report_markdown", "")
            if args.out:
                out_path = Path(args.out)
                _write_text_file(out_path, report_md)
                print(str(out_path))
            else:
                print(report_md)
            return

        # Should not reach here; argparse enforces subcommand
        _print_error("No command provided")
        sys.exit(2)

    except FileNotFoundError as e:
        _print_error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        _print_error(str(e))
        sys.exit(1)
    except Exception as e:
        # Defensive catch-all with concise error message
        _print_error(f"Unexpected error: {e.__class__.__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()