#!/usr/bin/env python3
"""
chunk_reduce.py: Local Map-Reduce Compression Pipeline for Ollama

Splits multi-megabyte files (logs, code dumps, traces) into overlapping chunks,
extracts high-signal information locally via Ollama (Map), and aggregates/deduplicates
the results into a concise brief (Reduce) for consumption by agy-cli.

License: GNU AGPLv3
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if not (src_dir / "ollama_bridge").exists():
    src_dir = Path("/data/agy_ollama_mcp/src")
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ollama_bridge.config import DEFAULT_FALLBACK_MODEL, OLLAMA_HOST
from ollama_bridge.engine import local_map_reduce_file, query_ollama, chunk_lines_overlap, map_worker
import concurrent.futures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked Map-Reduce compression pipeline for local Ollama instances."
    )
    parser.add_argument(
        "-f", "--file", help="Path to input file. If omitted, reads from stdin."
    )
    parser.add_argument(
        "-g",
        "--goal",
        required=True,
        help="Target extraction goal (e.g. 'Identify database connection timeouts and schema errors').",
    )
    parser.add_argument(
        "-m", "--model", default="", help=f"Ollama model name (default: auto-detected or {DEFAULT_FALLBACK_MODEL})."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Number of lines per chunk (default: 400).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping lines between consecutive chunks (default: 50).",
    )
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent chunk map requests to Ollama (default: 2).",
    )

    args = parser.parse_args()

    if args.file:
        summary = local_map_reduce_file(
            file_path=args.file,
            extraction_goal=args.goal,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            concurrency=args.concurrency,
            model=args.model,
        )
    else:
        if sys.stdin.isatty():
            sys.stderr.write("[!] Reading from standard input (Ctrl+D to finish)...\n")
        raw_text = sys.stdin.read()
        lines = raw_text.splitlines(keepends=True)
        if not lines:
            print("Input is empty.")
            return

        if len(lines) <= args.chunk_size:
            summary = query_ollama(
                f"GOAL: {args.goal}\n\nCONTENT:\n{''.join(lines)}",
                system="Provide a dense technical extraction matching the goal. Omit boilerplate.",
                model=args.model,
                temperature=0.1,
            )
        else:
            chunks = chunk_lines_overlap(lines, chunk_size=args.chunk_size, overlap=args.overlap)
            mapped: list[str] = [""] * len(chunks)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
                futures = {
                    executor.submit(map_worker, c, args.goal, args.model): idx
                    for idx, c in enumerate(chunks)
                }
                for f in concurrent.futures.as_completed(futures):
                    idx = futures[f]
                    res = f.result()
                    if "NO_SIGNAL" not in res and res.strip():
                        mapped[idx] = res

            filtered = [s for s in mapped if s.strip()]
            if not filtered:
                summary = f"No signal matching '{args.goal}' found across {len(chunks)} chunks."
            else:
                intermediate = "\n\n".join(
                    [f"### Findings Section {i + 1}\n{s}" for i, s in enumerate(filtered)]
                )
                reduce_prompt = (
                    f"PRIMARY OBJECTIVE: {args.goal}\n\n"
                    f"INTERMEDIATE CHUNK EXTRACTIONS:\n{intermediate}\n\n"
                    "TASK:\n"
                    "Synthesize these findings into a unified, actionable technical summary. "
                    "Highlight root causes, frequencies, key identifiers, and affected modules."
                )
                reduce_system = (
                    "You are an executive technical synthesizer. Consolidate extraction notes "
                    "into a cohesive summary. Deduplicate repeated items and highlight critical issues."
                )
                summary = query_ollama(reduce_prompt, system=reduce_system, model=args.model, temperature=0.1)

    print(summary)


if __name__ == "__main__":
    main()
