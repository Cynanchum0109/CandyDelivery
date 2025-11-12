#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import threading
import queue
import time
import multiprocessing as mp

EXPRESSIONS = {"neutral", "happy", "surprised", "sad"}


def draw_face(expression="neutral"):
    face = np.ones((400, 600, 3), dtype=np.uint8) * 255  # white background

    # Eyes
    left_eye_center = (200, 180)
    right_eye_center = (400, 180)
    eye_radius = 40
    color = (0, 0, 0)

    cv2.circle(face, left_eye_center, eye_radius, color, -1)
    cv2.circle(face, right_eye_center, eye_radius, color, -1)

    # Mouth
    if expression == "happy":
        cv2.ellipse(face, (300, 280), (80, 40), 0, 0, 180, color, 5)
    elif expression == "sad":
        cv2.ellipse(face, (300, 340), (80, 40), 0, 180, 360, color, 5)
    elif expression == "surprised":
        cv2.circle(face, (300, 300), 30, color, 5)
    else:
        cv2.line(face, (220, 320), (380, 320), color, 5)

    return face


def _run_loop(initial_expression, allow_keyboard, command_queue):
    current_expression = (
        initial_expression if initial_expression in EXPRESSIONS else "neutral"
    )

    try:
        cv2.namedWindow("TurtleBot Face", cv2.WINDOW_AUTOSIZE)
    except cv2.error as err:
        print(f"[FaceDisplay] Unable to create OpenCV window: {err}")
        print("[FaceDisplay] Falling back to headless mode; expressions will not be shown.")
        return

    print("Face display controls: 1-neutral, 2-happy, 3-surprised, 4-sad, q-quit")
    running = True
    while running:
        try:
            while True:
                new_expr = command_queue.get_nowait()
                if new_expr == "__quit__":
                    running = False
                    break
                if new_expr in EXPRESSIONS:
                    current_expression = new_expr
        except queue.Empty:
            pass

        try:
            img = draw_face(current_expression)
            cv2.imshow("TurtleBot Face", img)
        except cv2.error as err:
            print(f"[FaceDisplay] OpenCV display error: {err}")
            print("[FaceDisplay] Closing face window.")
            break

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if allow_keyboard:
            if key == ord('1'):
                current_expression = "neutral"
            elif key == ord('2'):
                current_expression = "happy"
            elif key == ord('3'):
                current_expression = "surprised"
            elif key == ord('4'):
                current_expression = "sad"

        if cv2.getWindowProperty("TurtleBot Face", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


class FaceDisplay:
    def __init__(self, initial_expression="neutral", allow_keyboard=False):
        self.expression = initial_expression if initial_expression in EXPRESSIONS else "neutral"
        self.allow_keyboard = allow_keyboard
        self._queue = mp.Queue()
        self._process = None

    def start(self, blocking=False):
        if blocking:
            _run_loop(self.expression, self.allow_keyboard, self._queue)
        else:
            self._process = mp.Process(
                target=_run_loop,
                args=(self.expression, self.allow_keyboard, self._queue),
                daemon=True,
            )
            self._process.start()

    def set_expression(self, expression):
        if expression in EXPRESSIONS and self._queue:
            self._queue.put(expression)

    def stop(self):
        if self._queue:
            self._queue.put("__quit__")
        if self._process and self._process.is_alive():
            self._process.join(timeout=1)


def main():
    display = FaceDisplay(allow_keyboard=True)
    display.start(blocking=True)


if __name__ == "__main__":
    main()
