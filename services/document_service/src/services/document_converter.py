"""PDF to Markdown conversion (service layer).

Saves an uploaded PDF into the room's storage folder, extracts plain text per page with
PyMuPDF (``fitz``) and writes a sibling ``.md`` file. No OCR and no layout/table
recovery: scanned PDFs yield no text and are rejected with 422.

Storage layout: ``<DOCUMENT_STORAGE_ROOT>/<chat_room_id>/<folder>/`` (default root
``<repo>/uxaag/storage/document_service``)
(shared with ``markdown_store.py``). Raises ``HTTPException`` directly, so errors map
straight to HTTP codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from shared.logger import get_logger

try:
    # PyMuPDF is optional at import time so the service can still boot; conversion then returns 503.
    import fitz
except ImportError:  # pragma: no cover - runtime dependency guard
    fitz = None


@dataclass
class ConversionResult:
    """Outcome of a PDF conversion.

    Attributes:
        markdown: Final Markdown text (with title/source header when added).
        markdown_path: Absolute server path of the generated ``.md`` file.
        source_pdf_path: Absolute server path of the stored original PDF.
    """
    markdown: str
    markdown_path: str
    source_pdf_path: str


class LightweightPdfMarkdownConverter:
    """Convert uploaded PDFs to Markdown with plain PyMuPDF text extraction.

    Uploads are streamed to disk, size-capped by ``MAX_PDF_UPLOAD_BYTES`` (default 5 MiB) and
    stored next to the generated ``.md`` under the chat room's folder.
    """

    def __init__(self) -> None:
        """Resolve the storage root, create it and read the upload size limit."""
        self.logger = get_logger("document-service")
        # File is src/services/document_converter.py, so parents[4] is the ``uxaag`` directory.
        self.project_root = Path(__file__).resolve().parents[4]
        default_root = self.project_root / "storage" / "document_service"
        # Must match ``MarkdownFileStore`` so PDFs and Markdown share one tree.
        self.storage_root = Path(os.getenv("DOCUMENT_STORAGE_ROOT") or default_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.max_upload_size_bytes = int(os.getenv("MAX_PDF_UPLOAD_BYTES", str(5 * 1024 * 1024)))
        # Stream uploads in 1 MiB chunks so large files are never fully held in memory.
        self.read_chunk_size = 1024 * 1024

    @staticmethod
    def _validate_chat_room_id(chat_room_id: int) -> int:
        """Return ``chat_room_id`` if positive.

        Raises:
            HTTPException: 400 if it is zero or negative.
        """
        if chat_room_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chat_room_id must be a positive integer.",
            )
        return chat_room_id

    def _room_root(self, chat_room_id: int) -> Path:
        """Return (and create) the room's storage folder ``<storage_root>/<chat_room_id>``."""
        room_root = self.storage_root / str(self._validate_chat_room_id(chat_room_id))
        room_root.mkdir(parents=True, exist_ok=True)
        return room_root

    @staticmethod
    def _normalize_folder_prefix(folder: str) -> str:
        """Normalize a user-supplied folder into a clean ``a/b`` relative path.

        Converts backslashes to slashes, strips outer slashes and rejects empty segments and ``..``.

        Raises:
            HTTPException: 400 if the folder is invalid.
        """
        cleaned = (folder or "").strip().replace("\\", "/").strip("/")
        if not cleaned:
            return ""
        for part in cleaned.split("/"):
            if not part or part == "..":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid folder path.",
                )
        return cleaned

    def _resolve_target_folder(self, chat_room_id: int, folder: str) -> Path:
        """Resolve and create the absolute upload folder inside the room root.

        Raises:
            HTTPException: 400 if the resolved path escapes the room root.
        """
        room_root = self._room_root(chat_room_id).resolve()
        prefix = self._normalize_folder_prefix(folder)
        target = room_root / prefix if prefix else room_root
        target = target.resolve()

        # Defense in depth against path traversal (symlinks etc.) after ``resolve()``.
        if target != room_root and room_root not in target.parents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid folder path.",
            )

        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _safe_stem(filename: str) -> str:
        """Reduce a filename to a filesystem-safe stem (alphanumerics, ``-``, ``_``; fallback ``document``)."""
        stem = Path(filename or "document").stem
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem) or "document"

    async def _persist_upload_to_disk(self, upload_file: UploadFile, target_path: Path) -> None:
        """Stream an upload to ``target_path``, enforcing the size cap.

        Raises:
            HTTPException: 413 if the upload exceeds ``max_upload_size_bytes`` (partial file is deleted).
        """
        total_size = 0
        with target_path.open("wb") as out_file:
            while True:
                chunk = await upload_file.read(self.read_chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > self.max_upload_size_bytes:
                    out_file.close()
                    target_path.unlink(missing_ok=True)
                    self.logger.warning(
                        "Upload rejected: reason=too_large limit_bytes=%d",
                        self.max_upload_size_bytes,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Max size exceeded. Maximum allowed PDF size is {self.max_upload_size_bytes} bytes.",
                    )
                out_file.write(chunk)
        # Rewind the upload so later readers of the same ``UploadFile`` start from the beginning.
        await upload_file.seek(0)
        self.logger.debug("PDF stored: bytes=%d file=%s", total_size, target_path.name)

    @staticmethod
    def _post_process_markdown(markdown: str, source_name: str, total_pages: int) -> str:
        """Clean raw page text and add a title header.

        Drops lines that are only a number (page numbers), collapses 3+ blank lines, fixes ATX
        headings missing a space after ``#``. Returns ``""`` for empty text. If the text already
        starts with a ``# `` heading it is returned as is, otherwise a header with the source file
        name and page count is prepended.
        """
        lines = markdown.splitlines()
        cleaned_lines: list[str] = []

        for line in lines:
            normalized = line.rstrip()
            if re.fullmatch(r"\s*\d+\s*", normalized):
                continue
            cleaned_lines.append(normalized)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", text, flags=re.MULTILINE)
        text = text.strip()

        if not text:
            return ""

        title = Path(source_name).stem.replace("_", " ").strip() or "PDF Document"
        header = f"# {title}\n\n- Source file: `{source_name}`\n- Pages: {total_pages}\n\n"
        if text.startswith("# "):
            return text + "\n"
        return header + text + "\n"

    def _count_pdf_pages(self, pdf_path: Path) -> int:
        """Return the PDF's page count (also validates that the file opens).

        Raises:
            HTTPException: 503 if PyMuPDF is missing, 400 if the PDF cannot be read.
        """
        if fitz is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyMuPDF is not installed. Install PyMuPDF to enable PDF conversion.",
            )

        try:
            with fitz.open(str(pdf_path)) as document:
                return int(document.page_count)
        except Exception as exc:
            self.logger.exception("PDF page count failed: file=%s", pdf_path.name)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to read PDF: {exc!s}",
            ) from exc

    def _extract_text_markdown(self, pdf_path: Path) -> str:
        """Extract text page by page as ``## Page N`` sections (blank pages are skipped).

        Raises:
            HTTPException: 503 if PyMuPDF is missing.
        """
        if fitz is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyMuPDF is not installed. Install PyMuPDF to enable PDF conversion.",
            )

        parts: list[str] = []
        with fitz.open(str(pdf_path)) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    parts.append(f"## Page {index}\n\n{text}")
        return "\n\n".join(parts).strip()

    def _convert_pdf_path_to_markdown(self, pdf_path: Path) -> str:
        """Run extraction and map any failure to HTTP 502.

        Raises:
            HTTPException: 502 if PyMuPDF fails (503 from the extractor is also wrapped as 502).
        """
        try:
            markdown = self._extract_text_markdown(pdf_path)
        except Exception as exc:
            self.logger.exception("PDF conversion failed: file=%s", pdf_path.name)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"PDF to markdown conversion failed: {exc!s}",
            ) from exc

        return str(markdown or "").strip()

    def _remove_stored_pdf(self, pdf_path: Path) -> None:
        """Best-effort delete of a stored PDF after a failed conversion; failures are only logged."""
        try:
            pdf_path.unlink(missing_ok=True)
            self.logger.info("Orphan PDF removed after failed conversion: file=%s", pdf_path.name)
        except OSError:
            self.logger.warning("Orphan PDF removal failed: file=%s", pdf_path.name, exc_info=True)

    async def convert_pdf_to_markdown(
        self,
        chat_room_id: int,
        upload_file: UploadFile,
        folder: str = "",
    ) -> ConversionResult:
        """Store an uploaded PDF and convert it to Markdown in the room folder.

        Args:
            chat_room_id: Target room (must be positive).
            upload_file: The uploaded PDF (filename must end with ``.pdf``).
            folder: Optional sub-folder relative to the room root.

        Returns:
            ``ConversionResult`` with the Markdown text and the absolute paths written.

        Raises:
            HTTPException: 400 (not a PDF, bad folder or room id, unreadable PDF), 413 (too
                large), 422 (no extractable text), 502 (conversion failed), 503 (PyMuPDF missing).
        """
        if not upload_file.filename or not upload_file.filename.lower().endswith(".pdf"):
            self.logger.warning(
                "Upload rejected: reason=not_pdf chat_room_id=%s", chat_room_id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported.",
            )

        started = time.perf_counter()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_name = self._safe_stem(upload_file.filename)
        # Timestamp + random suffix prevents collisions when the same file is uploaded twice.
        unique_name = f"{base_name}_{timestamp}_{uuid4().hex[:8]}"

        target_folder = self._resolve_target_folder(chat_room_id, folder)
        pdf_path = target_folder / f"{unique_name}.pdf"
        markdown_path = target_folder / f"{unique_name}.md"

        await self._persist_upload_to_disk(upload_file, pdf_path)
        try:
            total_pages = self._count_pdf_pages(pdf_path)
            raw_markdown = self._convert_pdf_path_to_markdown(pdf_path)
            markdown = self._post_process_markdown(raw_markdown, upload_file.filename, total_pages)

            if not markdown:
                self.logger.warning(
                    "PDF conversion rejected: reason=no_text chat_room_id=%s pages=%d",
                    chat_room_id,
                    total_pages,
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="PDF conversion completed but no text content was extracted.",
                )

            markdown_path.write_text(markdown, encoding="utf-8")
        except Exception:
            # Conversion failed after the PDF was stored: drop the orphan PDF (best effort).
            self._remove_stored_pdf(pdf_path)
            raise
        self.logger.info(
            "PDF converted chat_room_id=%s pages=%d duration_ms=%.0f",
            chat_room_id,
            total_pages,
            (time.perf_counter() - started) * 1000,
        )

        return ConversionResult(
            markdown=markdown,
            markdown_path=str(markdown_path),
            source_pdf_path=str(pdf_path),
        )


# Backward-compatible alias for older imports.
DocumentConverter = LightweightPdfMarkdownConverter
