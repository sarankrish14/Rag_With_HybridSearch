"""Semantic breakpoint detection and chunk building."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from semantic_chunker.logging_setup import logger
from semantic_chunker.settings import (
  MAX_CHUNK_SIZE,
  MIN_CHUNK_CHARS,
  MIN_CHUNK_SIZE,
  OVERLAP_SENTENCES,
)


def find_breakpoints(embeddings: np.ndarray) -> tuple[list[int], float]:
  breakpoints = [0]

  distances = []
  for i in range(len(embeddings) - 1):
    sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
    dist = 1.0 - sim
    distances.append((i + 1, dist))

  if not distances:
    return breakpoints, 0.0

  dist_values = [d for _, d in distances]
  mean = float(np.mean(dist_values))
  std = float(np.std(dist_values))
  auto_threshold = mean + 0.8 * std

  logger.info(
    f"Distance stats → min: {min(dist_values):.3f} | mean: {mean:.3f} | "
    f"max: {max(dist_values):.3f} | std: {std:.3f} | "
    f"auto threshold: {auto_threshold:.3f}"
  )

  for idx, dist in distances:
    if dist >= auto_threshold:
      breakpoints.append(idx)

  return breakpoints, auto_threshold


def build_chunks(
  sentences: list[str], breakpoints: list[int]
) -> tuple[list[list[str]], list[int], list[int]]:
  segments: list[list[str]] = []
  for i, start in enumerate(breakpoints):
    end = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(sentences)
    segments.append(sentences[start:end])

  merged: list[list[str]] = []
  merged_starts: list[int] = []
  buffer: list[str] = []
  buffer_start: int = 0

  cursor = 0
  for seg in segments:
    if not buffer:
      buffer_start = cursor
    buffer.extend(seg)
    cursor += len(seg)
    if len(buffer) >= MIN_CHUNK_SIZE:
      while len(buffer) > MAX_CHUNK_SIZE:
        merged.append(buffer[:MAX_CHUNK_SIZE])
        merged_starts.append(buffer_start)
        buffer_start += MAX_CHUNK_SIZE
        buffer = buffer[MAX_CHUNK_SIZE:]
      if buffer:
        merged.append(buffer)
        merged_starts.append(buffer_start)
      buffer = []
      buffer_start = cursor

  if buffer:
    if merged:
      combined = merged[-1] + buffer
      last_start = merged_starts[-1]
      merged.pop()
      merged_starts.pop()

      while len(combined) > MAX_CHUNK_SIZE:
        merged.append(combined[:MAX_CHUNK_SIZE])
        merged_starts.append(last_start)
        last_start += MAX_CHUNK_SIZE
        combined = combined[MAX_CHUNK_SIZE:]
      if combined:
        merged.append(combined)
        merged_starts.append(last_start)
    else:
      merged.append(buffer)
      merged_starts.append(buffer_start)

  if OVERLAP_SENTENCES > 0 and len(merged) > 1:
    overlapped: list[list[str]] = [merged[0]]
    original_lengths = [len(merged[0])]
    for i in range(1, len(merged)):
      tail = merged[i - 1][-OVERLAP_SENTENCES:]
      overlapped.append(tail + merged[i])
      original_lengths.append(len(merged[i]))
    return overlapped, original_lengths, merged_starts

  original_lengths = [len(c) for c in merged]
  return merged, original_lengths, merged_starts


def is_garbage_chunk(text: str) -> bool:
  return len(text.strip()) < MIN_CHUNK_CHARS
