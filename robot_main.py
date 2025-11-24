#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program for Candy Delivery Robot
Sequentially runs:
1. Human detection and approach (approach_human.py)
2. Voice conversation (voice_candy_chat.py)
"""

import sys
import time
import cv2
from ultralytics import YOLO
import subprocess
import threading
import queue
import numpy as np
from datetime import datetime

# Import voice chat components
from voice_candy_chat import VoiceCandyChat

# ========= CONFIGURE SSH INFO HERE =========
ROBOT_USER = "turtle-one"
ROBOT_IP = "10.5.12.184"
SSH = f"{ROBOT_USER}@{ROBOT_IP}"
# ===========================================

def send_vel(linear_x, angular_z=0.0):
    """Send velocity command to robot via SSH."""
    yaml_msg = (
        "{linear: {x: " + str(linear_x) + ", y: 0.0, z: 0.0}, "
        "angular: {x: 0.0, y: 0.0, z: " + str(angular_z) + "}}"
    )
    cmd = (
        f"ssh {SSH} "
        f"\"rostopic pub -1 /cmd_vel_mux/input/navi geometry_msgs/Twist "
        f"\\\"{yaml_msg}\\\"\""
    )
    subprocess.run(cmd, shell=True)


class HumanApproach:
    """Human detection and approach module."""
    
    def __init__(self, target_height=280, fps_target=30, camera_id=0, 
                 cap=None, video_writer=None):
        self.target_height = target_height
        self.fps_target = fps_target
        self.camera_id = camera_id
        self.running = False
        self.human_reached = False
        self.cap = cap  # Can be passed from main program
        self.model = None
        self.video_writer = video_writer  # Can be passed from main program
        self._own_cap = False  # Track if we own the camera
        self._own_writer = False  # Track if we own the writer
        
    def initialize(self):
        """Initialize camera and YOLO model."""
        print("=" * 60)
        print("Initializing Human Detection System...")
        print("=" * 60)
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        print("Model loaded successfully.")
        
        # Initialize camera if not provided
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            self._own_cap = True
            if not self.cap.isOpened():
                print("Error: could not open webcam.")
                return False
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def run(self, timeout=None):
        """
        Run human detection and approach.
        Returns True when human is reached, False on error or timeout.
        """
        if not self.initialize():
            return False
        
        self.running = True
        self.human_reached = False
        
        frame_queue = queue.Queue(maxsize=2)
        result_queue = queue.Queue(maxsize=2)
        
        def capture_frames():
            """Continuously capture frames."""
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Note: Video recording is handled by background thread in main()
                # No need to write here to avoid duplicate writes
                
                # Put frame in detection queue
                if not frame_queue.full():
                    frame_queue.put((time.time(), frame))
                else:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put((time.time(), frame))
                    except queue.Empty:
                        pass
        
        def process_detections():
            """Process frames with YOLO."""
            while self.running:
                try:
                    timestamp, frame = frame_queue.get(timeout=0.1)
                    results = self.model(frame, verbose=False, imgsz=640)
                    
                    max_height = 0
                    annotated_frame = frame.copy()
                    
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            if cls == 0:  # person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                h = y2 - y1
                                if h > max_height:
                                    max_height = h
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"Person: {h}px", (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if not result_queue.full():
                        result_queue.put((timestamp, annotated_frame, max_height))
                    else:
                        try:
                            result_queue.get_nowait()
                            result_queue.put((timestamp, annotated_frame, max_height))
                        except queue.Empty:
                            pass
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Detection error: {e}")
                    continue
        
        # Start threads
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        process_thread = threading.Thread(target=process_detections, daemon=True)
        capture_thread.start()
        process_thread.start()
        
        print("\nHuman follower started...")
        print("Approaching human. Press 'q' to skip to voice chat.")
        print("=" * 60)
        
        frame_time = 1.0 / self.fps_target
        last_display_time = time.time()
        start_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check timeout
                if timeout and (current_time - start_time) > timeout:
                    print(f"\nTimeout reached ({timeout}s). Moving to voice chat...")
                    break
                
                # Control display frame rate
                if current_time - last_display_time < frame_time:
                    time.sleep(0.01)
                    continue
                
                last_display_time = current_time
                
                try:
                    timestamp, annotated_frame, max_height = result_queue.get(timeout=0.1)
                    fps = 1.0 / (current_time - timestamp) if current_time > timestamp else 0
                    
                    # Add FPS display
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Movement logic
                    if max_height == 0:
                        status_text = "No human → STOP"
                        send_vel(0.0)
                    else:
                        if max_height < self.target_height:
                            status_text = f"Human detected (h={max_height}) → APPROACHING"
                            send_vel(-0.25)
                        else:
                            status_text = f"Human detected (h={max_height}) → REACHED!"
                            send_vel(0.0)
                            # Human reached - set flag and break
                            self.human_reached = True
                            print("\n" + "=" * 60)
                            print("Human reached! Transitioning to voice chat...")
                            print("=" * 60)
                            time.sleep(1.0)  # Brief pause before transition
                            break
                    
                    # Add status text
                    cv2.putText(annotated_frame, status_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Add recording indicator
                    if self.video_writer:
                        cv2.putText(annotated_frame, "REC", (640 - 80, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.circle(annotated_frame, (640 - 30, 30), 8, (0, 0, 255), -1)
                    
                    cv2.imshow("Human Detection - Approach Phase", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nSkipping to voice chat...")
                        break
                        
                except queue.Empty:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Waiting for detection...", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Human Detection - Approach Phase", blank_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Keep resources for continued recording
            self.cleanup(keep_resources=True)
        
        return self.human_reached
    
    def cleanup(self, keep_resources=False):
        """
        Clean up resources.
        If keep_resources=True, don't release camera and video writer (for continued recording).
        """
        self.running = False
        time.sleep(0.5)  # Give threads time to finish
        
        if not keep_resources:
            if self.video_writer and self._own_writer:
                self.video_writer.release()
                print("Video saved.")
            
            if self.cap and self._own_cap:
                self.cap.release()
        
        cv2.destroyAllWindows()
        send_vel(0.0)  # Stop robot
        print("Human detection system stopped.")


def record_video_continuously(cap, video_writer, running_flag):
    """Background thread to continuously record video."""
    while running_flag[0]:
        ret, frame = cap.read()
        if ret and video_writer:
            video_writer.write(frame)
        else:
            time.sleep(0.01)
        time.sleep(1.0 / 30.0)  # ~30 FPS


def main():
    """Main program flow."""
    print("=" * 60)
    print("Candy Delivery Robot - Main Program")
    print("=" * 60)
    print()
    
    # Initialize camera and video recording for entire session
    print("Initializing camera and video recording...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize video writer for entire session
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"robot_session_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print(f"Warning: Could not initialize video writer")
        video_writer = None
    else:
        print(f"Recording entire session to: {video_filename}")
    print()
    
    # Start continuous video recording in background
    recording_running = [True]
    recording_thread = threading.Thread(
        target=record_video_continuously, 
        args=(cap, video_writer, recording_running),
        daemon=True
    )
    recording_thread.start()
    
    try:
        # Step 1: Human Detection and Approach
        approach = HumanApproach(target_height=280, fps_target=30, camera_id=0,
                                cap=cap, video_writer=video_writer)
        human_reached = approach.run(timeout=300)  # 5 minute timeout
        
        if not human_reached:
            print("\nWarning: Human not reached. Proceeding to voice chat anyway...")
            time.sleep(2)
        
        # Step 2: Voice Conversation (video recording continues in background)
        print("\n" + "=" * 60)
        print("Starting Voice Conversation System...")
        print("(Video recording continues in background)")
        print("=" * 60)
        print()
        
        try:
            voice_chat = VoiceCandyChat()
            voice_chat.run()
        except Exception as e:
            print(f"Error in voice chat: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        # Stop recording
        recording_running[0] = False
        time.sleep(0.5)  # Give recording thread time to finish
        
        # Final cleanup
        if video_writer:
            video_writer.release()
            print(f"\nVideo saved to: {video_filename}")
        
        if cap:
            cap.release()
        
        send_vel(0.0)
        print("\n" + "=" * 60)
        print("Program completed. Robot stopped.")
        print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        send_vel(0.0)
        sys.exit(0)

