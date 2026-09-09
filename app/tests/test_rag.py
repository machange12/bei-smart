"""Run all 30 RAG test questions and print a full report.

For every question: the question itself, detected language, classified
intent, extracted entities, lookup status, and the final answer text --
so extraction failures are visible without reading any pipeline code.

Usage:
    python app/tests/test_rag.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.rag import pipeline  # noqa: E402
from tests.rag_questions import QUESTIONS  # noqa: E402


def main() -> None:
    print(f"AgriPulse RAG test run -- {len(QUESTIONS)} questions\n")
    print("=" * 100)

    status_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    errors = 0

    for i, (question, note) in enumerate(QUESTIONS, start=1):
        print(f"\n[{i:02d}] Q: {question}")
        print(f"     note: {note}")

        try:
            start = time.time()
            result = pipeline.answer(question)
            elapsed = time.time() - start
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"     ERROR: {exc}")
            traceback.print_exc()
            print("-" * 100)
            continue

        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        intent_counts[result["intent"]] = intent_counts.get(result["intent"], 0) + 1
        language_counts[result["language"]] = language_counts.get(result["language"], 0) + 1

        print(f"     language:  {result['language']}")
        print(f"     intent:    {result['intent']}")
        print(f"     entities:  {result['entities']}")
        print(f"     status:    {result['status']}")
        print(f"     confidence:{result['confidence']}")
        print(f"     sources:   {result['sources']}")
        print(f"     answer:    {result['answer']}")
        print(f"     ({elapsed:.2f}s)")
        print("-" * 100)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total questions: {len(QUESTIONS)}   Errors: {errors}")
    print(f"By intent:   {intent_counts}")
    print(f"By language: {language_counts}")
    print(f"By status:   {status_counts}")


if __name__ == "__main__":
    main()
