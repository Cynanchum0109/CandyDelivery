import asyncio
import websockets
import threading

class FaceServer:
    def __init__(self):
        self.websocket = None
        self.loop = None
        self._connected = threading.Event()
        self._exit_requested = threading.Event()  # 退出请求标志
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()

    async def handler(self, websocket):
        print("Face window connected.")
        self.websocket = websocket
        self._connected.set()  # 标记已连接
        try:
            async for message in websocket:
                # 接收来自前端的信息
                if message == "EXIT" or message == "exit":
                    print("Exit requested from web interface.")
                    self._exit_requested.set()
        except websockets.exceptions.ConnectionClosed:
            print("Face window disconnected.")
        finally:
            self._connected.clear()
    
    def is_exit_requested(self):
        """检查是否收到退出请求"""
        return self._exit_requested.is_set()
    
    def clear_exit_request(self):
        """清除退出请求标志（用于重置）"""
        self._exit_requested.clear()
    
    def is_connected(self):
        """检查 WebSocket 是否已连接"""
        return self._connected.is_set()
    
    def wait_for_connection(self, timeout=None):
        """等待 WebSocket 连接，返回是否连接成功"""
        return self._connected.wait(timeout=timeout)

    def start_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        
        async def run_server():
            async with websockets.serve(self.handler, "localhost", 8765):
                print("Face server running on ws://localhost:8765")
                await asyncio.Future()  # run forever
        
        self.loop.run_until_complete(run_server())

    async def send_state(self, state):
        if self.websocket:
            try:
                await self.websocket.send(state)
            except Exception as e:
                print(f"Error sending state: {e}")

    def send(self, state):
        # 使用服务器线程的事件循环发送消息
        if self.loop and self.loop.is_running():
            # 使用 call_soon_threadsafe 安全地从其他线程调用
            asyncio.run_coroutine_threadsafe(self.send_state(state), self.loop)
        else:
            # 如果循环还没启动，等待一下
            import time
            time.sleep(0.1)
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.send_state(state), self.loop)
