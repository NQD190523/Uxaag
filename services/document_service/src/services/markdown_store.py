"""Per-room Markdown file store on local disk (service layer).

Layout: ``<DOCUMENT_STORAGE_ROOT>/<chat_room_id>/...`` (default root
``<repo>/uxaag/storage/document_service``) - one folder per chat
room, containing arbitrary sub-folders and ``.md`` files (PDFs written by the converter
live there too but are hidden from the tree). Every user-supplied path is resolved and
checked to stay inside the room folder (no absolute paths, no ``..``). Errors are raised
as ``HTTPException``.

SECURITY: no per-user authorization here; anyone who can reach the service can touch any
room (see bug D1).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from fastapi import HTTPException, status


@dataclass(frozen=True)
class MarkdownFile:
    """Descriptor of a file or folder inside a room.

    Attributes:
        relative_path: POSIX-style path relative to the room root.
        absolute_path: Absolute path on the server disk.
        entry_type: ``file`` or ``folder``.
    """
    relative_path: str
    absolute_path: str
    entry_type: str


class MarkdownFileStore:
    """Simple, safe markdown file store for editor read/write on local disk."""

    def __init__(self) -> None:
        """Resolve the storage root and create it.

        The root is ``DOCUMENT_STORAGE_ROOT`` when set, else ``uxaag/storage/document_service``.
        """
        # File is src/services/markdown_store.py, so parents[4] is the ``uxaag`` directory.
        default_root = Path(__file__).resolve().parents[4] / "storage" / "document_service"
        # Must match ``LightweightPdfMarkdownConverter`` so PDFs and Markdown share one tree.
        self.root = Path(os.getenv("DOCUMENT_STORAGE_ROOT") or default_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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
        """Return (and create) the room's folder."""
        room_id = self._validate_chat_room_id(chat_room_id)
        room_root = self.root / str(room_id)
        room_root.mkdir(parents=True, exist_ok=True)
        return room_root

    def _resolve_relative(self, chat_room_id: int, relative_path: str) -> Path:
        """Resolve a room-relative path to an absolute path guaranteed to be inside the room folder.

        Raises:
            HTTPException: 400 if the path is empty, absolute, contains ``..`` or escapes the root.
        """
        if not relative_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="relative_path is required.",
            )

        rel = Path(relative_path)
        if rel.is_absolute():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Absolute paths are not allowed.",
            )

        if ".." in rel.parts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path traversal is not allowed.",
            )

        room_root = self._room_root(chat_room_id)
        full = (room_root / rel).resolve()
        # Second check after ``resolve()`` also catches symlink escapes.
        root_resolved = room_root.resolve()
        if root_resolved not in full.parents and full != root_resolved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path.",
            )

        return full

    def ensure_chat_room_folder(self, chat_room_id: int) -> str:
        """Create the room folder if needed and return its absolute path."""
        return str(self._room_root(chat_room_id))

    @staticmethod
    def _markdown_title(path: Path) -> str:
        """Return the display title of a Markdown file.

        Uses the ``title:`` key of a leading ``---`` front-matter block when present, else the
        file stem. Any read error falls back to the stem.
        """
        title = path.stem
        try:
            content = path.read_text(encoding="utf-8")
            if content.startswith("---"):
                for line in content.splitlines()[1:]:
                    if line.strip() == "---":
                        break
                    if line.lower().startswith("title:"):
                        parsed = line.split(":", 1)[1].strip().strip('"').strip("'")
                        if parsed:
                            return parsed
        except Exception:
            return title
        return title

    def _node_id(self, chat_room_id: int, relative_path: str) -> str:
        """Build a stable tree node id: ``<room_id>`` for the root, else ``<room_id>/<relative_path>``."""
        return f"{chat_room_id}" if not relative_path else f"{chat_room_id}/{relative_path}"

    def _build_tree_node(self, chat_room_id: int, room_root: Path, current: Path) -> dict:
        """Recursively build one tree node (dict matching ``schemas.FileNode``).

        Folders come before files (then case-insensitive by name); only sub-folders and ``.md``
        files are listed, so uploaded PDFs are not shown.
        """
        relative_path = "" if current == room_root else str(current.relative_to(room_root)).replace("\\", "/")
        stat = current.stat()

        if current.is_dir():
            children: list[dict] = []
            for item in sorted(current.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                if item.is_dir():
                    children.append(self._build_tree_node(chat_room_id, room_root, item))
                elif item.suffix.lower() == ".md":
                    children.append(self._build_tree_node(chat_room_id, room_root, item))
            return {
                "id": self._node_id(chat_room_id, relative_path),
                "name": str(chat_room_id) if current == room_root else current.name,
                "type": "folder",
                "path": relative_path,
                "children": children,
                "last_modified": stat.st_mtime,
                "size": 0,
            }

        return {
            "id": self._node_id(chat_room_id, relative_path),
            "name": self._markdown_title(current),
            "type": "file",
            "path": relative_path,
            "children": [],
            "last_modified": stat.st_mtime,
            "size": stat.st_size,
        }

    def build_tree(self, chat_room_id: int) -> dict:
        """Return the room's whole tree as a nested dict."""
        room_root = self._room_root(chat_room_id)
        return self._build_tree_node(chat_room_id, room_root, room_root)

    def list_entries(self, chat_room_id: int) -> list[MarkdownFile]:
        """Return every file and folder under the room (any extension), sorted by relative path."""
        room_root = self._room_root(chat_room_id)
        files: list[MarkdownFile] = []
        for p in room_root.rglob("*"):
            if p == room_root:
                continue
            if p.is_file() or p.is_dir():
                files.append(
                    MarkdownFile(
                        relative_path=str(p.relative_to(room_root)).replace("\\", "/"),
                        absolute_path=str(p),
                        entry_type="folder" if p.is_dir() else "file",
                    )
                )
        files.sort(key=lambda f: f.relative_path.lower())
        return files

    def read_markdown(self, chat_room_id: int, relative_path: str) -> str:
        """Read a Markdown file as UTF-8 text.

        Raises:
            HTTPException: 400 if the path is invalid or not ``.md``, 404 if missing.
        """
        path = self._resolve_relative(chat_room_id, relative_path)
        if path.suffix.lower() != ".md":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .md files are supported for read.",
            )
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Markdown file not found.",
            )
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _normalize_folder_prefix(folder: str) -> str:
        """Normalize a folder string into a clean ``a/b`` relative path ("" for the root).

        Raises:
            HTTPException: 400 on empty segments or ``..``.
        """
        cleaned = folder.strip().replace("\\", "/").strip("/")
        if not cleaned:
            return ""
        for part in cleaned.split("/"):
            if not part or part == "..":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid folder path.",
                )
        return cleaned

    def create_markdown_in_folder(
        self,
        chat_room_id: int,
        folder: str,
        filename: str,
        content: str = "",
    ) -> MarkdownFile:
        """Create a new Markdown file; never overwrites.

        Args:
            chat_room_id: Target room.
            folder: Parent folder relative to the room root ("" for the root).
            filename: Single file name ending in ``.md`` (no slashes).
            content: Initial text.

        Returns:
            Descriptor of the created file.

        Raises:
            HTTPException: 400 (bad filename/folder), 409 (something already exists there).
        """
        name = filename.strip()
        if "/" in name or "\\" in name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="filename must be a single name only; use folder for subdirectories.",
            )
        if not name.lower().endswith(".md"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="filename must end with .md",
            )

        prefix = self._normalize_folder_prefix(folder)
        relative = f"{prefix}/{name}" if prefix else name

        path = self._resolve_relative(chat_room_id, relative)
        room_root = self._room_root(chat_room_id)

        if path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A file or folder already exists at this path.",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "", encoding="utf-8")
        return MarkdownFile(
            relative_path=str(path.relative_to(room_root)).replace("\\", "/"),
            absolute_path=str(path),
            entry_type="file",
        )

    def write_markdown(self, chat_room_id: int, relative_path: str, content: str) -> MarkdownFile:
        """Create or overwrite a Markdown file (parent folders are created).

        Raises:
            HTTPException: 400 if the path is invalid or not ``.md``.
        """
        path = self._resolve_relative(chat_room_id, relative_path)
        room_root = self._room_root(chat_room_id)
        if path.suffix.lower() != ".md":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .md files are supported for write.",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "", encoding="utf-8")
        return MarkdownFile(
            relative_path=str(path.relative_to(room_root)).replace("\\", "/"),
            absolute_path=str(path),
            entry_type="file",
        )

    def create_folder(self, chat_room_id: int, relative_path: str) -> MarkdownFile:
        """Create a folder (and parents); succeeds if it already exists.

        Raises:
            HTTPException: 400 if the path is invalid.
        """
        path = self._resolve_relative(chat_room_id, relative_path)
        room_root = self._room_root(chat_room_id)
        path.mkdir(parents=True, exist_ok=True)
        return MarkdownFile(
            relative_path=str(path.relative_to(room_root)).replace("\\", "/"),
            absolute_path=str(path),
            entry_type="folder",
        )

    def delete_entry(self, chat_room_id: int, relative_path: str) -> None:
        """Delete a file, or a folder with all its contents (irreversible).

        Raises:
            HTTPException: 400 (room root or invalid path), 404 (not found).
        """
        path = self._resolve_relative(chat_room_id, relative_path)
        room_root = self._room_root(chat_room_id)
        if path.resolve() == room_root.resolve():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the chat room root folder.",
            )
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Path not found.",
            )
        if path.is_dir():
            # Delete deepest paths first (reverse order) so directories are empty when removed.
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
        else:
            path.unlink()

    def rename_entry(self, chat_room_id: int, from_relative: str, to_relative: str) -> MarkdownFile:
        """Rename or move a file or folder within the room tree.

        Args:
            chat_room_id: Target room.
            from_relative: Existing path relative to the room root.
            to_relative: New path relative to the room root (parents are created).

        Returns:
            Descriptor of the destination.

        Raises:
            HTTPException: 400 (room root / folder into its own descendant / invalid path),
                404 (source missing), 409 (destination exists).
        """
        src = self._resolve_relative(chat_room_id, from_relative)
        dst = self._resolve_relative(chat_room_id, to_relative)
        room_root = self._room_root(chat_room_id)

        if src.resolve() == room_root.resolve():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot rename the chat room root folder.",
            )

        if not src.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Path not found.",
            )

        if dst.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Target path already exists.",
            )

        if src.is_dir():
            try:
                # Success of ``relative_to`` means dst is inside src; the HTTPException is raised inside the
                # ``try`` and is not a ValueError, so it propagates. ``ValueError`` means dst is outside src.
                dst.resolve().relative_to(src.resolve())
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot move a folder into itself or its descendants.",
                )
            except ValueError:
                pass

        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

        return MarkdownFile(
            relative_path=str(dst.relative_to(room_root)).replace("\\", "/"),
            absolute_path=str(dst),
            entry_type="folder" if dst.is_dir() else "file",
        )

