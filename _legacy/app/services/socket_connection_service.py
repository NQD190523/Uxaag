from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        self.rooms: dict[str, List[WebSocket]] = {}  # room_id -> list of WS

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = []
        self.rooms[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.rooms:
            if websocket in self.rooms[room_id]:
                self.rooms[room_id].remove(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def broadcast(self, message: str, room_id: str):
        if room_id not in self.rooms:
            return
        for connection in self.rooms[room_id][:]:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection, room_id)