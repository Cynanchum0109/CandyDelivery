import threading
import tkinter as tk
from PIL import Image, ImageTk
import time
import queue


class ExpressionWindow(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cmd_queue = queue.Queue()

        # 表情图片路径
        self.images = {
            "idle": "faces/idle.png",
            "speaking": "faces/speaking.png",
            "listening": "faces/listening.png",
            "thinking": "faces/thinking.png",
        }

    def run(self):
        self.root = tk.Tk()
        self.root.title("Robot Face")
        self.root.geometry("400x400")

        # 加载初始表情
        self.label = tk.Label(self.root)
        self.label.pack(expand=True)

        # 更新循环
        self.update_loop()
        self.root.mainloop()

    def update_loop(self):
        try:
            while not self.cmd_queue.empty():
                name = self.cmd_queue.get_nowait()
                self._apply_expression(name)
        except:
            pass

        self.root.after(50, self.update_loop)

    def _apply_expression(self, name):
        if name not in self.images:
            return
        img = Image.open(self.images[name]).resize((350, 350))
        self.tk_img = ImageTk.PhotoImage(img)
        self.label.config(image=self.tk_img)

    def set_expression(self, name):
        self.cmd_queue.put(name)


# usage:
# from expression_window import ExpressionWindow
# self.face = ExpressionWindow()
# self.face.start()
# self.face.set_expression("idle")
