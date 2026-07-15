"""Run: python -m semantic_chunker [--test] [--force] [path/to/file.pdf]"""

import sys
from pathlib import Path

from semantic_chunker.display import print_results
from semantic_chunker.pipeline import semantic_chunk_pdf
from semantic_chunker.settings import PDF_PATH
from semantic_chunker.tests import run_tests


def main() -> None:
    if "--test" in sys.argv:
        run_tests()
        return

    force = "--force" in sys.argv
    pdf_args = [
        arg for arg in sys.argv[1:]
        if arg not in ("--test", "--force") and not arg.startswith("-")
    ]
    pdf_path = pdf_args[0] if pdf_args else PDF_PATH

    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    chunks = semantic_chunk_pdf(pdf_path, force=force)
    print_results(chunks)


if __name__ == "__main__":
    main()
