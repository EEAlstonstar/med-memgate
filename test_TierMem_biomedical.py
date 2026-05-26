#!/usr/bin/env python3
"""
TierMem test script for the Biomedical dataset.

Simulates multi-visit doctor-patient dialogues and tests memory retrieval
across 3 QA categories:
  cat=0  complete   – list all entities across all visits      (expected: R-path)
  cat=1  first_visit – entities from the first visit only      (expected: R-path)
  cat=2  latest      – entities from the most recent visit     (expected: S-path)

Prerequisites:
  1. Qdrant running:   ./start_qdrant.sh  (or docker)
  2. API key set:      export OPENAI_API_KEY=...
  3. Sessions file:    python scripts/biomedical/1_convert_to_sessions.py

Usage:
  # Quick smoke-test (10 sessions, no vLLM router)
  python test_TierMem_biomedical.py --limit 10 --router-type llm --max-workers 2

  # Full run with vLLM router
  python test_TierMem_biomedical.py \\
    --router-type vllm \\
    --router-base-url http://localhost:8000/v1 \\
    --router-model Qwen3-0.6B

  # Medical-enhanced BinaryRouter (no LLM router needed)
  python test_TierMem_biomedical.py --router-type binary --limit 50

  # Compare baseline vs medical prompt addendum
  python test_TierMem_biomedical.py --router-type openai --run-id bio_baseline
  python test_TierMem_biomedical.py --router-type openai --medical-addendum --run-id bio_medical
"""

import argparse
import sys
import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file=None, log_dir="logs"):
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
    else:
        log_path = None

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    if log_path:
        fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024,
                                 backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    logging.getLogger("src").setLevel(logging.INFO)
    return log_path


logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger("src").setLevel(logging.INFO)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from core.runner.run_benchmark_multi import run_benchmark_multi
from core.datasets import biomedical
from src.memory.linked_view_system import LinkedViewSystem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test med-memgate on the Biomedical multi-visit dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--executor", type=str, default="thread",
                        choices=["thread", "process"])

    router_group = parser.add_argument_group("Router")
    router_group.add_argument("--router-type", type=str, default="vllm",
                              choices=["openai", "vllm", "llm", "binary"])
    router_group.add_argument("--router-model", type=str, default="Qwen3-0.6B")
    router_group.add_argument("--router-base-url", type=str,
                              default="http://localhost:8000/v1")
    router_group.add_argument("--router-api-key", type=str, default="vllm-api-key")
    router_group.add_argument("--router-thinking", action="store_true", default=True)
    router_group.add_argument("--no-router-thinking", dest="router_thinking",
                              action="store_false")
    # Medical addendum flag – injects MEDICAL_ROUTER_ADDENDUM into LLM/vLLM router prompt
    router_group.add_argument("--medical-addendum", action="store_true", default=False,
                              help="Inject medical routing rules into the LLM router prompt")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return 1

    from datetime import datetime
    run_id = args.run_id or f"medmemgate_bio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = setup_logging(f"{run_id}.log")

    username = os.getenv("USER", "user")
    collection_name = args.collection_name or f"mem0_bio_{username}_{run_id}"

    router_config = {"type": args.router_type}
    if args.router_type == "vllm":
        router_config["base_url"] = args.router_base_url
        router_config["model"] = args.router_model
        router_config["api_key"] = args.router_api_key
        router_config["is_thinking_model"] = args.router_thinking
    elif args.router_type == "openai":
        router_config["model"] = args.router_model or args.model
        if args.router_api_key:
            router_config["api_key"] = args.router_api_key
        if args.router_base_url:
            router_config["base_url"] = args.router_base_url

    # Pass medical addendum flag into router config so LinkedViewSystem can pick it up
    if args.medical_addendum:
        from src.linked_view.prompts import MEDICAL_ROUTER_ADDENDUM
        router_config["domain_addendum"] = MEDICAL_ROUTER_ADDENDUM

    lv_cfg = {
        "benchmark_name": "biomedical",
        "mem0_config": {
            "backend": "mem0",
            "llm": {
                "provider": "openai",
                "config": {"model": args.model},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": args.qdrant_host,
                    "port": args.qdrant_port,
                    "collection_name": collection_name,
                },
            },
        },
        "memory_system_model": args.model,
        "router_config": router_config,
        "use_query_rewriter": False,
        "use_dual_retrieval": False,
        "top_k": 5,
        "max_research_iters": 3,
        "use_reranker": False,
        "ablation_bm25_only": False,
    }

    print(f"\n{'='*60}")
    print("med-memgate – Biomedical Benchmark")
    print(f"{'='*60}")
    print(f"Model      : {args.model}")
    print(f"Router     : {args.router_type}  medical-addendum={args.medical_addendum}")
    print(f"Limit      : {args.limit or 'all (270 sessions)'}")
    print(f"Workers    : {args.max_workers}")
    print(f"Run ID     : {run_id}")
    print(f"Collection : {collection_name}")
    print(f"{'='*60}\n")

    try:
        system = LinkedViewSystem(lv_cfg)
        print("med-memgate system created.")
    except Exception as e:
        print(f"Failed to create system: {e}")
        import traceback; traceback.print_exc()
        return 1

    try:
        summary = run_benchmark_multi(
            system=system,
            dataset_module=biomedical,
            benchmark_name="biomedical",
            run_id=run_id,
            config={"model_name": args.model, "split": "test"},
            output_dir=args.output_dir,
            limit=args.limit,
            max_workers=args.max_workers,
            executor_type=args.executor,
            system_config=lv_cfg,
            load_only=False,
            qa_max_workers=1,
        )

        result_path = f"{args.output_dir}/biomedical/linked_view/{run_id}/"
        print(f"\nResults saved to: {result_path}")
        print("Metrics:")
        for k, v in summary.get("metrics", {}).items():
            print(f"  {k}: {v}")
        return 0

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
