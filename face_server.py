import asyncio
import websockets
import threading

class FaceServer:
    def __init__(self):
        self.websocket = None
        threading.Thread(target=self.start_server, daemon=True).start()

    async def handler(self, websocket):
        print("Face window connected.")
        self.websocket = websocket
        while True:
            await asyncio.sleep(0.1)

    def start_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        server = websockets.serve(self.handler, "localhost", 8765)
        loop.run_until_complete(server)
        print("Face server running on ws://localhost:8765")
        loop.run_forever()

    async def send_state(self, state):
        if self.websocket:
            await self.websocket.send(state)

    def send(self, state):
        asyncio.run(self.send_state(state))
