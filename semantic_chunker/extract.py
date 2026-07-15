"""PDF text extraction via Docling."""

import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from semantic_chunker.logging_setup import logger


def extract_text_from_pdf(pdf_path: str) -> tuple[str, list[dict]]:
  try:
    logger.info("Using Docling for structured extraction...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
      }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    page_map: list[dict] = []
    full_parts: list[str] = []

    for element, _level in doc.iterate_items():
      page_no = None
      if hasattr(element, "prov") and element.prov:
        page_no = element.prov[0].page_no if element.prov else None

      if isinstance(element, TableItem):
        table_text = _table_to_text(element)
        if table_text:
          full_parts.append(table_text)
          page_map.append({"page_no": page_no, "text": table_text, "type": "table"})
        continue

      text = element.text if hasattr(element, "text") else None
      if not text or len(text.strip()) < 5:
        continue

      cleaned = _clean_text(text)
      if cleaned:
        full_parts.append(cleaned)
        page_map.append({"page_no": page_no, "text": cleaned, "type": "text"})

    full_text = "\n".join(full_parts)
    logger.info(f"Extracted {len(full_text)} chars across {len(page_map)} elements")
    return full_text, page_map

  except FileNotFoundError:
    logger.error(f"PDF not found: '{pdf_path}' — check the PDF_PATH setting")
    raise
  except Exception as e:
    logger.error(f"Docling extraction failed: {e}")
    raise


def _clean_text(text: str) -> str:
  text = re.sub(r"-\n(\w)", r"\1", text)
  text = re.sub(r"\s+", " ", text).strip()
  if re.fullmatch(r"\d{1,4}", text):
    return ""
  return text


def _table_to_text(table_item) -> str:
  try:
    if not hasattr(table_item, "data") or table_item.data is None:
      return ""

    rows = []
    for row in table_item.data.grid:
      cell_texts = [cell.text.strip() if cell.text else "" for cell in row]
      if any(cell_texts):
        rows.append(" | ".join(cell_texts))

    if not rows:
      return ""

    return "\n".join(rows)

  except Exception as e:
    logger.debug(f"Table extraction failed, skipping: {e}")
    return ""
