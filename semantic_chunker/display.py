"""Terminal output for chunking results."""

import config
from semantic_chunker.settings import LOG_FILE


def print_results(chunks: list[dict]) -> None:
  print("\n" + "=" * 80)
  print(f"✅  SEMANTIC CHUNKING COMPLETE — {len(chunks)} chunks created")
  print("=" * 80)

  for c in chunks:
    print(f"\n{'─' * 80}")
    chunk_type = c.get("chunk_type", "text")
    print(
      f"CHUNK {c['chunk_index']}  "
      f"[{chunk_type} | {c['sentence_count']} sentences | "
      f"{c['char_count']} chars | page {c['page_no']}]"
    )
    print(f"ID: {c['chunk_id']}")
    print(f"{'─' * 80}")
    print(c["text"])

  total_sentences = sum(c["sentence_count"] for c in chunks)
  text_chunks = sum(1 for c in chunks if c.get("chunk_type", "text") == "text")
  table_chunks = sum(1 for c in chunks if c.get("chunk_type") == "table")
  print("\n" + "=" * 80)
  postgres_status = "enabled" if config.DATABASE_URL else "not configured"
  print(
    f"📈 Summary: {len(chunks)} chunks ({text_chunks} text, {table_chunks} table) | "
    f"{total_sentences} sentences | Postgres: {postgres_status} | log → {LOG_FILE}"
  )
  print("=" * 80)
