"""Run retrieval pipeline: python -m retrieval [--demo] [--serve]"""

import asyncio
import logging
import sys

import config


async def _run_queries(queries: list[str]) -> None:
    from retrieval.pipeline import RAGPipeline, _LOG_FORMAT, _get_logger, load_config

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    demo_logger = _get_logger("demo")

    pipeline = await RAGPipeline.create(load_config())
    try:
        for q in queries:
            demo_logger.info("=" * 60)
            result = await pipeline.arun(q, use_cache=False)
            print(result.answer)
    finally:
        await pipeline.aclose()


def main() -> None:
    if "--serve" in sys.argv:
        try:
            import uvicorn
            from retrieval.pipeline import app
        except ImportError as exc:
            print(f"FastAPI/uvicorn required for --serve: {exc}")
            sys.exit(1)

        uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            log_level="info",
        )
        return

    if "--demo" in sys.argv:
        test_queries = [
            "What is the main topic of the uploaded documents?",
            "hi",
            "What is semantic chunking?",
        ]
    else:
        query_args = [
            arg for arg in sys.argv[1:]
            if arg not in ("--demo", "--serve") and not arg.startswith("-")
        ]
        if not query_args:
            print("Usage: python -m retrieval [--demo] [--serve] [query ...]")
            sys.exit(1)
        test_queries = [" ".join(query_args)]

    asyncio.run(_run_queries(test_queries))


if __name__ == "__main__":
    main()
