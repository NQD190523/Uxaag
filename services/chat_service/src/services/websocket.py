"""WebSocket connection registry (service layer): tracks sockets per room and
broadcasts text frames.

State is in-memory and per process; cross-worker delivery of chat-list updates is
handled by ``redis_client.run_room_pubsub_listener``.
"""

from typing import Dict, Set, Tuple

from fastapi import WebSocket

from shared.logger import get_logger

logger = get_logger("chat-service")

class WebSocketService:
    """Registry of live WebSocket connections keyed by room id."""

    def __init__(self):
        # room_id (normalized string) -> set of WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    def _room_key(self, room_id: str) -> str:
        """Normalize path param so all handlers use the same dict key."""
        return str(room_id).strip()

    async def connect(self, websocket: WebSocket, room_id: str):
        """Register an already-accepted socket in the room.

        The router calls ``websocket.accept()`` before auth so rejections can send a
        close code; accepting twice would raise, so it is not done here.

        Args:
            websocket: The incoming connection (must already be accepted).
            room_id: Room id from the URL path.
        """
        key = self._room_key(room_id)

        if key not in self.active_connections:
            self.active_connections[key] = set()

        self.active_connections[key].add(websocket)
        logger.info(f"Client connected to room {key} | Total: {len(self.active_connections[key])}")

    def disconnect(self, websocket: WebSocket, room_id: str):
        """Unregister a socket; drop the room entry when its last socket leaves.

        Safe to call more than once for the same socket.
        """
        key = self._room_key(room_id)
        if key in self.active_connections:
            self.active_connections[key].discard(websocket)

            if not self.active_connections[key]:
                del self.active_connections[key]
                logger.info(f"Room {key} is now empty and removed")
            else:
                logger.info(f"Client disconnected from room {key} | Remaining: {len(self.active_connections[key])}")

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
            logger.warning(f"Broadcast skipped: Room {key!r} has no active connections")
            return

        # Make a copy to avoid modification during iteration
        connections = list(self.active_connections[key])
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to client in room {key}: {e}")
                disconnected.append(connection)

        # Clean up dead connections
        for conn in disconnected:
            self.disconnect(conn, key)

        n = len(connections) - len(disconnected)
        logger.info(f"Broadcast to {n} client(s) in room {key}")

    async def broadcast_all(self, message: str):
        """Send a text frame to every socket in every room on this process.

        Used for chat-list updates received from Redis pub/sub, so all open chat
        lists refresh regardless of which room each client is viewing.
        """
        total_sent = 0
        # Iterate over a snapshot because broadcast() may delete dead rooms.
        for room_id in list(self.active_connections.keys()):
            before = len(self.active_connections.get(room_id, set()))
            await self.broadcast(message=message, room_id=room_id)
            after = len(self.active_connections.get(room_id, set()))
            # Approximate delivered count (sockets that survived the send).
            total_sent += after if after <= before else before
        logger.info(f"Global broadcast delivered to {total_sent} client(s)")


websocket_service = WebSocketService()