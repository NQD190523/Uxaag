"""WebSocket connection registry (service layer): tracks sockets per room and
broadcasts text frames.

State is in-memory and per process; cross-worker delivery of chat-list updates is
handled by ``redis_client.run_room_pubsub_listener``, which sends them only to
members of the updated room (``services/room_list_fanout.py``).
"""

from typing import Dict, Optional, Set, Tuple

from fastapi import WebSocket

from shared.logger import get_logger

logger = get_logger("chat-service")

class WebSocketService:
    """Registry of live WebSocket connections keyed by room id."""

    def __init__(self):
        # room_id (normalized string) -> set of WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # socket -> user id (string) of the authenticated user who opened it. Lets
        # cross-room updates go only to users allowed to see them (security review).
        self.socket_users: Dict[WebSocket, str] = {}

    def _room_key(self, room_id: str) -> str:
        """Normalize path param so all handlers use the same dict key."""
        return str(room_id).strip()

    async def connect(self, websocket: WebSocket, room_id: str, user_id: Optional[str] = None):
        """Register an already-accepted socket in the room.

        The router calls ``websocket.accept()`` before auth so rejections can send a
        close code; accepting twice would raise, so it is not done here.

        Args:
            websocket: The incoming connection (must already be accepted).
            room_id: Room id from the URL path.
            user_id: Authenticated user behind the socket. A socket registered without
                it never receives cross-room updates (``broadcast_to_users``).
        """
        key = self._room_key(room_id)

        if key not in self.active_connections:
            self.active_connections[key] = set()

        self.active_connections[key].add(websocket)
        if user_id is not None:
            self.socket_users[websocket] = str(user_id)
        logger.debug("Socket registered room_id=%s total=%d", key, len(self.active_connections[key]))

    def disconnect(self, websocket: WebSocket, room_id: str):
        """Unregister a socket; drop the room entry when its last socket leaves.

        Safe to call more than once for the same socket.
        """
        key = self._room_key(room_id)
        # A socket belongs to exactly one room, so its user entry goes with it.
        self.socket_users.pop(websocket, None)
        if key in self.active_connections:
            self.active_connections[key].discard(websocket)

            if not self.active_connections[key]:
                del self.active_connections[key]
                logger.debug("Room has no sockets left room_id=%s", key)
            else:
                logger.debug("Socket unregistered room_id=%s remaining=%d", key, len(self.active_connections[key]))

    async def broadcast(self, message: str, room_id: str):
        """Send a text frame to every socket in one room.

        Sockets that fail to send are removed from the registry. A room with no
        sockets is skipped with a warning.

        Args:
            message: Already-serialized JSON string (see ``ws_payloads``).
            room_id: Target room id.
        """
        key = self._room_key(room_id)
        if key not in self.active_connections:
            logger.debug("Broadcast skipped: no sockets room_id=%s", key)
            return

        # Make a copy to avoid modification during iteration
        connections = list(self.active_connections[key])
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning("Socket send failed, dropping it room_id=%s error=%s", key, type(e).__name__)
                disconnected.append(connection)

        # Clean up dead connections
        for conn in disconnected:
            self.disconnect(conn, key)

        n = len(connections) - len(disconnected)
        logger.debug("Broadcast done room_id=%s delivered=%d", key, n)

    def has_connections(self) -> bool:
        """Return True if this process holds at least one open socket."""
        return bool(self.active_connections)

    async def broadcast_to_users(self, message: str, user_ids: Set[str]) -> int:
        """Send a text frame to every socket, in any room, owned by one of ``user_ids``.

        Used for chat-list updates (``last_message_updated``): a user sees the update
        in whichever room they have open, but only if they are a member of the room
        the update is about. SECURITY: this replaced a send-to-every-socket broadcast
        that leaked message text across rooms (security review, 2026-11-26).

        Args:
            message: Already-serialized JSON string.
            user_ids: Allowed recipients (user ids as strings).

        Returns:
            Number of sockets the frame was sent to.
        """
        delivered = 0
        dead: list[Tuple[WebSocket, str]] = []
        # Snapshot: sends await, and connects/disconnects may change the dicts meanwhile.
        for room_key, sockets in list(self.active_connections.items()):
            for socket in list(sockets):
                if self.socket_users.get(socket) not in user_ids:
                    continue
                try:
                    await socket.send_text(message)
                    delivered += 1
                except Exception as e:
                    logger.warning("Socket send failed, dropping it room_id=%s error=%s", room_key, type(e).__name__)
                    dead.append((socket, room_key))
        for socket, room_key in dead:
            self.disconnect(socket, room_key)
        logger.debug("Member broadcast done delivered=%d", delivered)
        return delivered


websocket_service = WebSocketService()