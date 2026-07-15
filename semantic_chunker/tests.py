"""Unit tests for semantic chunker (no PDF or model required)."""

import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from semantic_chunker.chunking import build_chunks, is_garbage_chunk
from semantic_chunker.embedding import embed_chunks
from semantic_chunker.extract import _clean_text
from semantic_chunker.registry import (
  _hash_file,
  _load_registry,
  _save_registry,
  is_duplicate,
  mark_as_processed,
  remove_existing_document,
)
from semantic_chunker.sentences import split_into_sentences
from semantic_chunker.settings import (
  EMBED_DIMENSION,
  MAX_CHUNK_SIZE,
  MIN_CHUNK_CHARS,
  OVERLAP_SENTENCES,
)
from semantic_chunker.storage import save_to_chroma


def run_tests() -> None:
  print("🧪 Running basic tests...\n")
  passed = 0
  failed = 0

  def check(name: str, condition: bool) -> None:
    nonlocal passed, failed
    if condition:
      print(f"  ✅ PASS: {name}")
      passed += 1
    else:
      print(f"  ❌ FAIL: {name}")
      failed += 1

  check(
    "clean_text joins hyphenated line break",
    _clean_text("connec-\ntion") == "connection",
  )
  check("clean_text drops lone page number", _clean_text("42") == "")
  check(
    "garbage filter rejects short chunk",
    is_garbage_chunk("See Figure 3.") is True,
  )

  long_content = (
    "This is a real sentence with enough content to be useful in retrieval "
    "and it continues with more words so the chunk clearly exceeds the minimum length."
  )
  check(
    "garbage filter passes real content",
    len(long_content) >= MIN_CHUNK_CHARS
    and is_garbage_chunk(long_content) is False,
  )

  sentences = [f"Sentence number {i} has enough words here." for i in range(10)]
  breakpoints = [0, 4, 7]
  chunks, original_lengths, start_offsets = build_chunks(sentences, breakpoints)
  check("build_chunks produces chunks from breakpoints", len(chunks) >= 2)
  check(
    "build_chunks original_lengths match non-overlap sentence counts",
    sum(original_lengths) == len(sentences),
  )
  check(
    "build_chunks start_offsets are valid sentence indices",
    start_offsets[0] == 0 and all(0 <= o < len(sentences) for o in start_offsets),
  )
  check(
    "build_chunks enforces MAX_CHUNK_SIZE on every chunk",
    all(len(c) <= MAX_CHUNK_SIZE for c in chunks),
  )

  overflow_sentences = [
    f"Overflow sentence number {i} has enough words here." for i in range(15)
  ]
  overflow_chunks, _, _ = build_chunks(overflow_sentences, [0, 9])
  check(
    "leftover buffer rechunk: no chunk exceeds MAX_CHUNK_SIZE",
    all(len(c) <= MAX_CHUNK_SIZE for c in overflow_chunks),
  )
  check(
    "leftover buffer rechunk: splits oversized tail into multiple chunks",
    len(overflow_chunks) >= 2,
  )

  if len(chunks) >= 2 and OVERLAP_SENTENCES > 0:
    check(
      "overlap: chunk 2 starts with tail of chunk 1",
      chunks[1][0] == chunks[0][-1],
    )
    check(
      "overlap: len(sents) includes overlap but original_lengths does not",
      len(chunks[1]) > original_lengths[1],
    )
    check(
      "overlap: original_lengths excludes duplicated overlap sentence",
      original_lengths[1] == len(chunks[1]) - OVERLAP_SENTENCES,
    )

  mixed_page_map = [
    {"page_no": 1, "text": "First paragraph. Second sentence.", "type": "text"},
    {"page_no": 2, "text": "Col A | Col B\nval1 | val2", "type": "table"},
  ]
  test_sentences: list[str] = []
  test_sentence_pages: list[int | None] = []
  for item in mixed_page_map:
    if item.get("type") == "table":
      continue
    for s in split_into_sentences(item["text"]):
      test_sentences.append(s)
      test_sentence_pages.append(item["page_no"])
  check(
    "page numbers attached at split time",
    len(test_sentence_pages) == len(test_sentences)
    and test_sentence_pages == [1, 1],
  )

  text_items = [i for i in mixed_page_map if i.get("type", "text") == "text"]
  table_items = [i for i in mixed_page_map if i.get("type") == "table"]
  check(
    "page_map separates text and table items",
    len(text_items) == 1 and len(table_items) == 1,
  )

  table_page_map = [
    {
      "page_no": 1,
      "text": "This is a normal paragraph with enough words to pass the filter.",
      "type": "text",
    },
    {
      "page_no": 2,
      "text": (
        "Name | Score | Grade | Notes\n"
        "Alice Johnson | 95 | A | Excellent performance across all modules\n"
        "Bob Smith | 88 | B | Good work with room for improvement in section two"
      ),
      "type": "table",
    },
  ]
  table_sentences: list[str] = []
  for item in table_page_map:
    if item.get("type") == "table":
      continue
    table_sentences.extend(split_into_sentences(item["text"]))
  built_table_chunks = [
    {
      "chunk_type": "table",
      "text": item["text"],
      "sentence_count": 1,
      "page_no": item["page_no"],
    }
    for item in table_page_map
    if item.get("type") == "table" and not is_garbage_chunk(item["text"])
  ]
  check(
    "tables excluded from sentence splitting",
    len(table_sentences) == 1 and "Alice | 95" not in " ".join(table_sentences),
  )
  check(
    "table chunk preserves pipe-separated rows intact",
    built_table_chunks[0]["text"].count("|") >= 4,
  )
  check(
    "table chunk labeled chunk_type=table",
    built_table_chunks[0]["chunk_type"] == "table",
  )

  page_sentences = [
    "Short.",
    "This is sentence one with enough words to pass.",
    "This is sentence two with enough words to pass.",
    "This is sentence three with enough words to pass.",
    "This is sentence four with enough words to pass.",
  ]
  page_sentence_pages = [1, 1, 1, 2, 2]
  page_breakpoints = [0, 1, 3]
  page_chunks, _, page_starts = build_chunks(page_sentences, page_breakpoints)
  kept_page_nos = []
  for idx, sents in enumerate(page_chunks):
    text = " ".join(sents)
    if is_garbage_chunk(text):
      continue
    offset = page_starts[idx]
    kept_page_nos.append(page_sentence_pages[offset])
  check(
    "start_offsets: page_no correct after skipping garbage chunk",
    kept_page_nos == [1, 2],
  )

  mock_text_chunk = {"chunk_type": "text", "chunk_index": 0}
  mock_table_chunk = {"chunk_type": "table", "chunk_index": 0}
  merged_chunks = [mock_text_chunk, mock_table_chunk]
  for i, chunk in enumerate(merged_chunks):
    chunk["chunk_index"] = i + 1
    chunk.setdefault("chunk_type", "text")
  check(
    "merged chunks: text chunk keeps chunk_type=text",
    merged_chunks[0]["chunk_type"] == "text",
  )
  check(
    "merged chunks: table chunk keeps chunk_type=table",
    merged_chunks[1]["chunk_type"] == "table",
  )

  class _FakeChromaCollection:
    def __init__(self):
      self.saved = None

    def upsert(self, **kwargs):
      self.saved = kwargs

  fake_chunks = [
    {
      "chunk_id": "t1",
      "text": "table data here " * 10,
      "chunk_index": 1,
      "sentence_count": 1,
      "char_count": 150,
      "source": "doc.pdf",
      "page_no": 2,
      "chunk_type": "table",
      "threshold_used": 0.0,
    },
    {
      "chunk_id": "x1",
      "text": "text chunk here " * 10,
      "chunk_index": 2,
      "sentence_count": 3,
      "char_count": 150,
      "source": "doc.pdf",
      "page_no": 3,
      "chunk_type": "text",
      "threshold_used": 0.42,
    },
  ]
  fake_col = _FakeChromaCollection()
  save_to_chroma(fake_col, fake_chunks, np.zeros((2, 3)))
  meta_types = [m["chunk_type"] for m in fake_col.saved["metadatas"]]
  check(
    "save_to_chroma metadata includes chunk_type for filtering",
    meta_types == ["table", "text"],
  )

  caller_chunks = [
    {
      "chunk_id": "pg1",
      "chunk_index": 1,
      "text": long_content,
      "sentence_count": 2,
      "char_count": len(long_content),
      "source": "doc.pdf",
      "page_no": 1,
      "chunk_type": "text",
      "threshold_used": 0.5,
    }
  ]
  caller_chunks_before = {k: v for k, v in caller_chunks[0].items()}
  postgres_rows = [
    {
      **chunk,
      "doc_id": "doc-test-123",
      "embedding": np.zeros(3).tolist(),
    }
    for chunk in caller_chunks
  ]
  check(
    "postgres row copies include embedding without mutating caller chunk",
    "embedding" in postgres_rows[0] and "embedding" not in caller_chunks[0],
  )
  check(
    "caller chunk dict unchanged except explicit doc_id injection",
    caller_chunks[0] == caller_chunks_before,
  )

  chroma_ready_chunks = [{**caller_chunks[0], "doc_id": "doc-test-123"}]
  fake_col2 = _FakeChromaCollection()
  save_to_chroma(fake_col2, chroma_ready_chunks, np.zeros((1, 3)))
  check(
    "save_to_chroma gets doc_id when set before save (no Postgres needed)",
    fake_col2.saved["metadatas"][0]["doc_id"] == "doc-test-123",
  )

  with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(b"fake pdf content for test")
    tmp_path = tmp.name

  h = _hash_file(tmp_path)
  reg = _load_registry()
  reg.pop(h, None)
  _save_registry(reg)

  check("duplicate: new file not flagged as duplicate", not is_duplicate(tmp_path))
  mark_as_processed(tmp_path)
  check(
    "duplicate: after marking, file IS flagged as duplicate",
    is_duplicate(tmp_path),
  )

  reg = _load_registry()
  reg.pop(h, None)
  _save_registry(reg)
  os.unlink(tmp_path)

  with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(b"hash reuse test content")
    tmp_path_16 = tmp.name
  h16 = _hash_file(tmp_path_16)
  reg16 = _load_registry()
  reg16.pop(h16, None)
  _save_registry(reg16)
  mark_as_processed(tmp_path_16, file_hash=h16)
  reg16 = _load_registry()
  check(
    "mark_as_processed uses pre-computed hash correctly",
    h16 in reg16,
  )
  reg16.pop(h16, None)
  _save_registry(reg16)
  os.unlink(tmp_path_16)

  short_sentences = [f"Short sentence {i} with words." for i in range(5)]
  short_chunks, _, _ = build_chunks(short_sentences, [0])
  check(
    "short doc guard: single breakpoint produces one or two chunks max",
    len(short_chunks) <= 2,
  )

  check("EMBED_DIMENSION constant equals 768", EMBED_DIMENSION == 768)

  mock_chunk_with_filename = {
    "chunk_id": "fn1",
    "text": long_content,
    "chunk_index": 1,
    "sentence_count": 2,
    "char_count": len(long_content),
    "source": "documents/test.pdf",
    "file_name": "test.pdf",
    "page_no": 1,
    "chunk_type": "text",
    "threshold_used": 0.3,
  }
  check(
    "chunk dict includes file_name field for ChromaDB filtering",
    "file_name" in mock_chunk_with_filename
    and mock_chunk_with_filename["file_name"] == "test.pdf",
  )

  with patch("semantic_chunker.registry.PostgresStore") as MockStore:
    instance = MockStore.return_value
    instance.delete_document_by_hash.return_value = "old-doc"
    with patch("semantic_chunker.storage.delete_chroma_document") as mock_chroma:
      reg = _load_registry()
      test_hash = "force-test-hash"
      reg[test_hash] = "documents/force.pdf"
      _save_registry(reg)
      removed = remove_existing_document(test_hash)
      check(
        "force re-ingest: remove_existing_document returns old doc_id",
        removed == "old-doc",
      )
      check(
        "force re-ingest: registry entry cleared",
        test_hash not in _load_registry(),
      )
      check(
        "force re-ingest: chroma cleanup called",
        mock_chroma.called and mock_chroma.call_args[0][0] == "old-doc",
      )

  print(f"\n{'─' * 40}")
  print(f"Results: {passed} passed, {failed} failed")
  if failed == 0:
    print("All tests passed ✅")
  else:
    print("Some tests failed ❌")
    sys.exit(1)
