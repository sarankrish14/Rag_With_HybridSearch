"""Sentence tokenization with NLTK."""

import re

import nltk
from nltk.tokenize import sent_tokenize


def _ensure_nltk_punkt() -> None:
  try:
    nltk.data.find("tokenizers/punkt_tab")
  except LookupError:
    nltk.download("punkt_tab", quiet=True)


def split_into_sentences(text: str) -> list[str]:
  _ensure_nltk_punkt()
  text = re.sub(r"\s+", " ", text).strip()
  sentences = sent_tokenize(text)
  return [s.strip() for s in sentences if len(s.strip()) > 10]
