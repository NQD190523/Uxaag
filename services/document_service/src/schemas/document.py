"""Pydantic request/response models for document_service (schema layer).

Every mutating response carries the refreshed ``FileNode`` tree so the UI can re-render
without a second request. ``FileNode`` is self-referential, hence the forward references
and the ``model_rebuild()`` calls at the bottom.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PdfToMarkdownResponse(BaseModel):
    """Response of the PDF conversion endpoint (saved PDF path, Markdown path and text, tree)."""
    success: bool
    chat_room_id: int
    source_pdf_path: str
    markdown_path: str
    markdown: str
    tree: "FileNode"


class MarkdownFileInfo(BaseModel):
    """One entry of the flat file listing; ``entry_type`` is ``file`` or ``folder``."""
    relative_path: str
    entry_type: str


class ListMarkdownFilesResponse(BaseModel):
    """Response of the flat listing endpoint."""
    success: bool
    chat_room_id: int
    files: list[MarkdownFileInfo]


class ReadMarkdownResponse(BaseModel):
    """Response of the read endpoint: file content as UTF-8 text."""
    success: bool
    chat_room_id: int
    relative_path: str
    content: str


class SaveMarkdownRequest(BaseModel):
    """Request body for saving a Markdown file (full content, overwrites)."""
    content: str


class SaveMarkdownResponse(BaseModel):
    """Response of the save endpoint, with the updated tree."""
    success: bool
    chat_room_id: int
    relative_path: str
    tree: "FileNode"


class InitChatRoomFolderResponse(BaseModel):
    """Response of the init-folder endpoint (server-side folder path plus tree)."""
    success: bool
    chat_room_id: int
    folder_path: str
    tree: "FileNode"


class CreateFolderResponse(BaseModel):
    """Response of the create-folder endpoint, with the updated tree."""
    success: bool
    chat_room_id: int
    relative_path: str
    tree: "FileNode"


class DeleteEntryResponse(BaseModel):
    """Response of the delete endpoint, with the updated tree."""
    success: bool
    chat_room_id: int
    relative_path: str
    tree: "FileNode"


class CreateMarkdownFileRequest(BaseModel):
    """Create a new .md file under an optional parent folder (relative to chat room root)."""

    folder: str = Field(
        default="",
        description="Parent folder path under room root, e.g. Hello or projects/docs (no leading slash)",
    )
    filename: str = Field(..., min_length=1, description="File name only, must end with .md")
    content: str = Field(default="", description="Initial markdown body")


class CreateMarkdownFileResponse(BaseModel):
    """Response of the create-file endpoint, with the updated tree."""
    success: bool
    chat_room_id: int
    relative_path: str
    tree: "FileNode"


class RenameEntryRequest(BaseModel):
    """Request body for rename/move; both paths are relative to the chat room root."""
    from_path: str = Field(..., description="Current path relative to chat room root")
    to_path: str = Field(..., description="New path relative to chat room root")


class RenameEntryResponse(BaseModel):
    """Response of the rename endpoint (requested source path, final path, tree)."""
    success: bool
    chat_room_id: int
    from_path: str
    to_path: str
    tree: "FileNode"


class FileNode(BaseModel):
    """Recursive tree node for a chat room folder.

    ``type`` is ``folder`` or ``file``; ``path`` is relative to the room root (empty for the
    root); ``name`` of a file is its front-matter ``title`` when present; ``last_modified``
    is a POSIX timestamp and ``size`` is in bytes (0 for folders).
    """
    id: str
    name: str
    type: str
    path: str
    children: list["FileNode"] = Field(default_factory=list)
    last_modified: float = 0
    size: int = 0


# Resolve the string forward references declared above (needed with ``from __future__ import annotations``).
FileNode.model_rebuild()
PdfToMarkdownResponse.model_rebuild()
SaveMarkdownResponse.model_rebuild()
InitChatRoomFolderResponse.model_rebuild()
CreateFolderResponse.model_rebuild()
DeleteEntryResponse.model_rebuild()
CreateMarkdownFileResponse.model_rebuild()
RenameEntryResponse.model_rebuild()

