"""Document service HTTP endpoints (router layer).

Thin handlers over ``MarkdownFileStore`` (per-room Markdown tree on disk) and
``LightweightPdfMarkdownConverter`` (PDF to Markdown). Mounted by ``main.py`` under
``/api/v1``, so the full path is ``/api/v1/documents/...``. The API gateway exposes it as
``/documents/...`` (``services.yaml`` entry ``documents``, ``protected: true``).

SECURITY: every room endpoint depends on ``require_room_member``: the gateway-injected
``x-user-id`` header is required (422 if missing) and the caller must be an active
participant of ``chat_room_id``, checked with chat-service (403 if not, 503 if chat-service
cannot answer; bug D1).
Errors are raised as ``HTTPException`` by the service layer (400/404/409/413/422/502/503).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile
from shared.logger import get_logger

from api.dependencies import require_room_member
from schemas.document import (
    CreateFolderResponse,
    CreateMarkdownFileRequest,
    CreateMarkdownFileResponse,
    DeleteEntryResponse,
    FileNode,
    InitChatRoomFolderResponse,
    ListMarkdownFilesResponse,
    MarkdownFileInfo,
    PdfToMarkdownResponse,
    ReadMarkdownResponse,
    RenameEntryRequest,
    RenameEntryResponse,
    SaveMarkdownRequest,
    SaveMarkdownResponse,
)
from services.document_converter import LightweightPdfMarkdownConverter
from services.markdown_store import MarkdownFileStore

# Module-level singletons: both are stateless apart from the shared on-disk storage root.
router = APIRouter(prefix="/documents", tags=["documents"])
converter = LightweightPdfMarkdownConverter()
md_store = MarkdownFileStore()
logger = get_logger("document-service")


def _room_tree(chat_room_id: int) -> FileNode:
    """Build the room's current file tree as a ``FileNode`` (returned by every mutating endpoint)."""
    return FileNode.model_validate(md_store.build_tree(chat_room_id))


@router.post("/chat-room/{chat_room_id}/convert-pdf", response_model=PdfToMarkdownResponse)
async def convert_pdf(
    chat_room_id: int,
    file: UploadFile = File(...),
    folder: str = Form(default=""),
    x_user_id: str = Depends(require_room_member),
):
    """POST /documents/chat-room/{chat_room_id}/convert-pdf - convert an uploaded PDF to Markdown.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: multipart form with ``file`` (required, ``.pdf``) and ``folder`` (optional target
    folder relative to the room root).
    Returns: ``PdfToMarkdownResponse`` (paths of the saved PDF and generated ``.md``, the
    Markdown text and the updated tree).
    Errors: 400 (not a PDF / bad folder / bad room id / unreadable PDF), 413 (file larger than
    ``MAX_PDF_UPLOAD_BYTES``), 422 (no text extracted), 502 (converter failed), 503 (PyMuPDF
    missing).
    """
    # Outcome and duration are logged by the converter ("PDF converted ...").
    result = await converter.convert_pdf_to_markdown(
        chat_room_id=chat_room_id,
        upload_file=file,
        folder=folder,
    )
    return PdfToMarkdownResponse(
        success=True,
        chat_room_id=chat_room_id,
        source_pdf_path=result.source_pdf_path,
        markdown_path=result.markdown_path,
        markdown=result.markdown,
        tree=_room_tree(chat_room_id),
    )



@router.post("/chat-room/{chat_room_id}/init-folder", response_model=InitChatRoomFolderResponse)
async def init_chat_room_folder(chat_room_id: int, x_user_id: str = Depends(require_room_member)):
    """POST /documents/chat-room/{chat_room_id}/init-folder - create the room's storage folder.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: none (idempotent).
    Returns: ``InitChatRoomFolderResponse`` with the folder path and the tree.
    Errors: 400 if ``chat_room_id`` is not positive.
    """
    folder_path = md_store.ensure_chat_room_folder(chat_room_id)
    logger.info("Room folder ready: chat_room_id=%s", chat_room_id)
    return InitChatRoomFolderResponse(
        success=True,
        chat_room_id=chat_room_id,
        folder_path=folder_path,
        tree=_room_tree(chat_room_id),
    )



@router.get("/chat-room/{chat_room_id}/files", response_model=ListMarkdownFilesResponse)
async def list_markdown_files(chat_room_id: int, x_user_id: str = Depends(require_room_member)):
    """GET /documents/chat-room/{chat_room_id}/files - flat list of every entry in the room.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Returns: ``ListMarkdownFilesResponse``. The list includes all
    files and folders (not only ``.md``), unlike ``/tree``.
    Errors: 400 if ``chat_room_id`` is not positive.
    """
    files = md_store.list_entries(chat_room_id)
    return ListMarkdownFilesResponse(
        success=True,
        chat_room_id=chat_room_id,
        files=[MarkdownFileInfo(relative_path=f.relative_path, entry_type=f.entry_type) for f in files],
    )



@router.get("/chat-room/{chat_room_id}/tree", response_model=FileNode)
async def get_folder_tree(chat_room_id: int, x_user_id: str = Depends(require_room_member)):
    """GET /documents/chat-room/{chat_room_id}/tree - nested folder/file tree of the room.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Returns: ``FileNode`` (folders plus ``.md`` files only).
    Errors: 400 if ``chat_room_id`` is not positive.
    """
    tree = md_store.build_tree(chat_room_id)
    return FileNode.model_validate(tree)



@router.get("/chat-room/{chat_room_id}/files/{relative_path:path}", response_model=ReadMarkdownResponse)
async def read_markdown_file(
    chat_room_id: int,
    relative_path: str,
    x_user_id: str = Depends(require_room_member),
):
    """GET /documents/chat-room/{chat_room_id}/files/{relative_path} - read one Markdown file.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Returns: ``ReadMarkdownResponse`` with the file content.
    Errors: 400 (not ``.md`` / path traversal / absolute path), 404 (file missing).
    """
    content = md_store.read_markdown(chat_room_id, relative_path)
    return ReadMarkdownResponse(
        success=True,
        chat_room_id=chat_room_id,
        relative_path=relative_path,
        content=content,
    )



@router.put("/chat-room/{chat_room_id}/files/{relative_path:path}", response_model=SaveMarkdownResponse)
async def save_markdown_file(
    chat_room_id: int,
    relative_path: str,
    request: SaveMarkdownRequest,
    x_user_id: str = Depends(require_room_member),
):
    """PUT /documents/chat-room/{chat_room_id}/files/{relative_path} - create or overwrite a file.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: ``SaveMarkdownRequest`` (``content``).
    Returns: ``SaveMarkdownResponse`` with the saved path and the updated tree.
    Errors: 400 (not ``.md`` / invalid path). Missing parent folders are created.
    """
    saved = md_store.write_markdown(chat_room_id, relative_path, request.content)
    # Path and size only; the Markdown content itself is never logged.
    logger.info(
        "File saved: chat_room_id=%s path=%s chars=%d",
        chat_room_id,
        saved.relative_path,
        len(request.content),
    )
    return SaveMarkdownResponse(
        success=True,
        chat_room_id=chat_room_id,
        relative_path=saved.relative_path,
        tree=_room_tree(chat_room_id),
    )



@router.post("/chat-room/{chat_room_id}/folders/{relative_path:path}", response_model=CreateFolderResponse)
async def create_folder(
    chat_room_id: int,
    relative_path: str,
    x_user_id: str = Depends(require_room_member),
):
    """POST /documents/chat-room/{chat_room_id}/folders/{relative_path} - create a folder.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: none (idempotent, parents are created).
    Returns: ``CreateFolderResponse`` with the created path and the updated tree.
    Errors: 400 (invalid path).
    """
    created = md_store.create_folder(chat_room_id, relative_path)
    logger.info("Folder created: chat_room_id=%s path=%s", chat_room_id, created.relative_path)
    return CreateFolderResponse(
        success=True,
        chat_room_id=chat_room_id,
        relative_path=created.relative_path,
        tree=_room_tree(chat_room_id),
    )


@router.post("/chat-room/{chat_room_id}/markdown-files", response_model=CreateMarkdownFileResponse)
async def create_markdown_file(
    chat_room_id: int,
    request: CreateMarkdownFileRequest,
    x_user_id: str = Depends(require_room_member),
):
    """POST /documents/chat-room/{chat_room_id}/markdown-files - create a new ``.md`` file.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: ``CreateMarkdownFileRequest`` (``folder`` relative to the
    room root, ``filename`` ending in ``.md``, optional ``content``).
    Returns: ``CreateMarkdownFileResponse`` with the new path and the updated tree.
    Errors: 400 (bad filename/folder), 409 (path already exists).
    """
    created = md_store.create_markdown_in_folder(
        chat_room_id,
        folder=request.folder,
        filename=request.filename,
        content=request.content,
    )
    logger.info("File created: chat_room_id=%s path=%s", chat_room_id, created.relative_path)
    return CreateMarkdownFileResponse(
        success=True,
        chat_room_id=chat_room_id,
        relative_path=created.relative_path,
        tree=_room_tree(chat_room_id),
    )


@router.patch("/chat-rooms/{chat_room_id}/rename", response_model=RenameEntryResponse)
@router.patch("/chat-room/{chat_room_id}/rename", response_model=RenameEntryResponse)
async def rename_entry(
    chat_room_id: int,
    request: RenameEntryRequest,
    x_user_id: str = Depends(require_room_member),
):
    """PATCH /documents/chat-room(s)/{chat_room_id}/rename - rename or move a file or folder.

    Registered under both ``chat-room`` and ``chat-rooms`` (the plural is kept for older clients).
    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Body: ``RenameEntryRequest`` (``from_path``, ``to_path``).
    Returns: ``RenameEntryResponse`` with the final path and the updated tree.
    Errors: 400 (room root / move into itself / invalid path), 404 (source missing),
    409 (target exists).
    """
    renamed = md_store.rename_entry(chat_room_id, request.from_path, request.to_path)
    logger.info(
        "Entry renamed: chat_room_id=%s from=%s to=%s",
        chat_room_id,
        request.from_path,
        renamed.relative_path,
    )
    return RenameEntryResponse(
        success=True,
        chat_room_id=chat_room_id,
        from_path=request.from_path,
        to_path=renamed.relative_path,
        tree=_room_tree(chat_room_id),
    )


@router.delete("/chat-room/{chat_room_id}/entries/{relative_path:path}", response_model=DeleteEntryResponse)
async def delete_entry(
    chat_room_id: int,
    relative_path: str,
    x_user_id: str = Depends(require_room_member),
):
    """DELETE /documents/chat-room/{chat_room_id}/entries/{relative_path} - delete a file or folder.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member (403 otherwise).
    Folders are deleted recursively and this cannot be undone.
    Returns: ``DeleteEntryResponse`` with the deleted path and the updated tree.
    Errors: 400 (room root / invalid path), 404 (path missing).
    """
    md_store.delete_entry(chat_room_id, relative_path)
    logger.info("Entry deleted: chat_room_id=%s path=%s", chat_room_id, relative_path)
    return DeleteEntryResponse(
        success=True,
        chat_room_id=chat_room_id,
        relative_path=relative_path,
        tree=_room_tree(chat_room_id),
    )


@router.get("/health")
async def document_router_health():
    """GET /documents/health - router-level health probe (``/api/v1/documents/health``).

    Auth: none. Returns: status, service name, feature tag and a UTC timestamp.
    """
    return {
        "status": "healthy",
        "service": "document-service",
        "feature": "pymupdf-text-converter",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
