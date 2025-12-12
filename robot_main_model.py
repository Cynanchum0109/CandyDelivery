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
import webbrowser
from pathlib import Path

# Import voice chat components
from voice_candy_chat_model import VoiceCandyChat
from face_server import FaceServer

from feature_processor_live import FeatureProcessorLive   # NEW

# ========= CONFIGURE SSH INFO HERE =========
ROBOT_USER = "turtle-one"
ROBOT_IP = "10.5.12.184"
SSH = f"{ROBOT_USER}@{ROBOT_IP}"
# ===========================================

def send_vel(linear_x, angular_z=0.0):
    """Send velocity command to robot via SSH (single command)."""
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




def move_backward(duration=5.0, velocity=0.25, update_interval=0.5):
    """
    Move robot backward for specified duration.
    
    Args:
        duration: Total time to move backward (seconds)
        velocity: Backward velocity (positive value, default 0.25)
        update_interval: How often to send velocity command (seconds)
    """
    print("\n" + "=" * 60)
    print("Moving backward to initial position...")
    print("=" * 60)
    
    start_time = time.time()
    end_time = start_time + duration
    
    try:
        while time.time() < end_time:
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            
            # Send backward velocity command
            send_vel(-velocity)
            print(f"Moving backward... {remaining:.1f}s remaining")
            
            # Sleep for update interval, but don't exceed remaining time
            sleep_time = min(update_interval, remaining)
            time.sleep(sleep_time)
        
        # Stop robot
        send_vel(0.0)
        print("✓ Reached initial position. Robot stopped.")
        print("=" * 60)
        
    except Exception as e:
        print(f"⚠ Error during backward movement: {e}")
        send_vel(0.0)  # Ensure robot stops on error


class HumanApproach:
    """Human detection and approach module."""
    
    def __init__(self, target_height=280, fps_target=30, camera_id=0, 
                 cap=None, video_writer=None, move_duration=0.5):
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
        self.move_duration = move_duration  # Duration to maintain movement (seconds)
        self.last_move_time = 0
        self.current_velocity = 0.0
        self.approach_start_time = None  # Track when approach started
        self.approach_duration = 0.0  # Total time spent approaching
        self._continuous_move_thread = None  # Thread for continuous movement
        self._continuous_move_stop = threading.Event()  # Event to stop continuous movement
        self._continuous_move_interval = 0.1  # Interval for sending commands during continuous move (seconds)
        self._target_velocity_for_continuous = 0.0  # Target velocity for continuous movement
        
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
    
    def _continuous_move_worker(self, velocity, duration):
        """
        Worker thread that continuously sends velocity commands for specified duration.
        This ensures the robot keeps moving even if the main loop is busy.
        """
        start_time = time.time()
        end_time = start_time + duration
        
        while not self._continuous_move_stop.is_set() and time.time() < end_time:
            send_vel(velocity)
            time.sleep(self._continuous_move_interval)
        
        # Stop the robot when done
        send_vel(0.0)
        self._continuous_move_stop.clear()
    
    def _start_continuous_move(self, velocity, duration):
        """Start continuous movement in a background thread."""
        # Stop any existing continuous movement
        self._stop_continuous_move()
        
        # Start new continuous movement
        self._continuous_move_stop.clear()
        self._target_velocity_for_continuous = velocity
        self._continuous_move_thread = threading.Thread(
            target=self._continuous_move_worker,
            args=(velocity, duration),
            daemon=True
        )
        self._continuous_move_thread.start()
    
    def _stop_continuous_move(self):
        """Stop continuous movement."""
        if self._continuous_move_thread and self._continuous_move_thread.is_alive():
            self._continuous_move_stop.set()
            self._continuous_move_thread.join(timeout=0.5)
        self._continuous_move_thread = None
        self._target_velocity_for_continuous = 0.0
    
    def run(self, timeout=None):
        """
        Run human detection and approach.
        Returns True when human is reached, False on error or timeout.
        """
        if not self.initialize():
            return False
        
        self.running = True
        self.human_reached = False
        self.approach_start_time = time.time()  # Record start time
        
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
                    best_box_info = None  # Store (height, y1, y2) of the best detection
                    annotated_frame = frame.copy()
                    frame_height = frame.shape[0]
                    
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            if cls == 0:  # person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                h = y2 - y1
                                if h > max_height:
                                    max_height = h
                                    best_box_info = (h, y1, y2, frame_height)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"Person: {h}px", (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if not result_queue.full():
                        result_queue.put((timestamp, annotated_frame, max_height, best_box_info))
                    else:
                        try:
                            result_queue.get_nowait()
                            result_queue.put((timestamp, annotated_frame, max_height, best_box_info))
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
                    queue_result = result_queue.get(timeout=0.1)
                    # Handle both old format (3 items) and new format (4 items)
                    if len(queue_result) == 3:
                        timestamp, annotated_frame, max_height = queue_result
                        best_box_info = None
                    else:
                        timestamp, annotated_frame, max_height, best_box_info = queue_result
                    
                    fps = 1.0 / (current_time - timestamp) if current_time > timestamp else 0
                    
                    # Add FPS display
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Movement logic with position-aware detection and continuous movement
                    time_since_last_move = current_time - self.last_move_time
                    
                    if max_height == 0:
                        # No human detected - stop immediately
                        if self.current_velocity != 0.0:
                            status_text = "No human → STOPPING"
                            # Stop continuous movement
                            self._stop_continuous_move()
                            send_vel(0.0)
                            self.current_velocity = 0.0
                            self.last_move_time = current_time
                        else:
                            status_text = "No human → STOP"
                    else:
                        # Check if human is actually close enough
                        # Criteria: height should be in reasonable range AND person should be in lower part of screen
                        height_ok = self.target_height <= max_height <= 400  # Reasonable height range
                        position_ok = True
                        
                        if best_box_info:
                            h, y1, y2, frame_height = best_box_info
                            # Person should be in lower 70% of screen (y2 > frame_height * 0.3)
                            # This ensures person is not too far away (which would show them in upper part)
                            position_ok = y2 > frame_height * 0.3
                            
                            # Additional check: if person is very tall (>400px) but in upper screen, keep approaching
                            if max_height > 400 and y1 < frame_height * 0.2:
                                height_ok = False  # Person is too far, keep approaching
                        
                        if not height_ok or not position_ok:
                            # Keep approaching: either height not in range, or person too high on screen
                            target_velocity = 0.25
                            
                            if max_height < self.target_height:
                                status_text = f"Human detected (h={max_height}) → APPROACHING (too small)"
                            elif max_height > 400:
                                status_text = f"Human detected (h={max_height}) → APPROACHING (too large/far)"
                            else:
                                status_text = f"Human detected (h={max_height}) → APPROACHING (position check)"
                            
                            # Start continuous movement if velocity changed or if move duration has elapsed
                            if self.current_velocity != target_velocity or time_since_last_move >= self.move_duration:
                                # Start continuous movement for move_duration seconds
                                self._start_continuous_move(target_velocity, self.move_duration)
                                self.current_velocity = target_velocity
                                self.last_move_time = current_time
                        else:
                            # Human reached: height in range AND position is good
                            status_text = f"Human detected (h={max_height}) → REACHED!"
                            if self.current_velocity != 0.0:
                                # Stop continuous movement
                                self._stop_continuous_move()
                                send_vel(0.0)
                                self.current_velocity = 0.0
                                self.last_move_time = current_time
                            # Human reached - set flag and break
                            self.human_reached = True
                            # Calculate total approach duration
                            if self.approach_start_time:
                                self.approach_duration = time.time() - self.approach_start_time
                            print("\n" + "=" * 60)
                            print("Human reached! Transitioning to voice chat...")
                            if self.approach_duration > 0:
                                print(f"Approach duration: {self.approach_duration:.1f}s")
                            print("=" * 60)
                            time.sleep(1.0)  # Brief pause before tfransition
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
            # Stop any continuous movement
            self._stop_continuous_move()
            # Calculate total approach duration if not already calculated
            if self.approach_start_time and self.approach_duration == 0.0:
                self.approach_duration = time.time() - self.approach_start_time
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


def open_face_window():
    """
    Open face.html automatically in the default browser.
    Assumes face.html is in the same directory as this file.
    Returns True if successful, False otherwise.
    """
    try:
        base_dir = Path(__file__).resolve().parent
        html_path = (base_dir / "face.html").resolve()
        if html_path.exists():
            url = html_path.as_uri()
            webbrowser.open(url, new=1)  # new tab / window
            print(f"✓ Opened face UI at {url}")
            return True
        else:
            print("⚠ Warning: face.html not found; cannot auto-open UI.")
            return False
    except Exception as e:
        print(f"⚠ Could not open face.html automatically: {e}")
        return False


def initialize_face_system():
    """
    Initialize FaceServer and open face window.
    Returns FaceServer instance if successful, None otherwise.
    """
    print("\n" + "=" * 60)
    print("Initializing Face Display System...")
    print("=" * 60)
    
    # Start FaceServer first
    print("Starting FaceServer (WebSocket server)...")
    try:
        face_server = FaceServer()
        # Give server time to start
        time.sleep(0.5)
        print("✓ FaceServer started on ws://localhost:8765")
    except Exception as e:
        print(f"⚠ Error starting FaceServer: {e}")
        return None
    
    # Open face window
    print("Opening face window in browser...")
    if open_face_window():
        # Wait a bit for browser to connect
        print("Waiting for browser connection...")
        if face_server.wait_for_connection(timeout=3):
            print("✓ Face window connected!")
        else:
            print("⚠ Face window opened but not connected yet (will retry later)")
    else:
        print("⚠ Could not open face window")
    
    print("=" * 60)
    print()
    return face_server


def record_video_continuously(cap, video_writer, running_flag, fps=30):
    """Background thread to continuously record video at specified FPS."""
    frame_interval = 1.0 / fps  # Time between frames
    last_frame_time = time.time()
    
    while running_flag[0]:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        # Only read and write frame if enough time has passed (to maintain FPS)
        if elapsed >= frame_interval:
            ret, frame = cap.read()
            if ret and video_writer:
                video_writer.write(frame)
                last_frame_time = current_time
            else:
                time.sleep(0.01)
        else:
            # Sleep for the remaining time until next frame
            time.sleep(max(0.001, frame_interval - elapsed))


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
    recording_fps = 30  # Target FPS for recording
    recording_thread = threading.Thread(
        target=record_video_continuously, 
        args=(cap, video_writer, recording_running, recording_fps),
        daemon=True
    )
    recording_thread.start()
    
    # Initialize face system before approaching human
    face_server = initialize_face_system()
    
    try:
        # Step 1: Human Detection and Approach
        approach = HumanApproach(target_height=280, fps_target=30, camera_id=0,
                                cap=cap, video_writer=video_writer, move_duration=2.0)
        human_reached = approach.run(timeout=300)  # 5 minute timeout
        
        if not human_reached:
            print("\n⚠ Warning: Human not reached. Proceeding to voice chat anyway...")
            time.sleep(2)
        
        # Step 2: Voice Conversation (video recording continues in background)
        print("\n" + "=" * 60)
        print("Starting Voice Conversation System...")
        print("(Video recording continues in background)")
        print("=" * 60)
        print()
        
        interaction_completed = False
        try:
            # Reuse existing face_server and don't open window again

            feature_stream = FeatureProcessorLive(cap, use_prediction=True)  
            feature_stream.start()

            voice_chat = VoiceCandyChat(
                open_face_window=False,
                face_server=face_server,
                feature_processor=feature_stream   # NEW
            )

            voice_chat.run()
            interaction_completed = True
        except Exception as e:
            print(f"⚠ Error in voice chat: {e}")
            import traceback
            traceback.print_exc()
            interaction_completed = True  # Still try to move back even on error
        
        # Step 3: Move backward to initial position after interaction
        if interaction_completed:
            # Use approach duration if available, otherwise use default duration
            backward_duration = approach.approach_duration if approach.approach_duration > 0 else 5.0
            # Add a small buffer (10%) to ensure we get back to initial position
            backward_duration = backward_duration * 1.1
            # Cap at reasonable maximum (30 seconds) and minimum (2 seconds)
            backward_duration = max(2.0, min(backward_duration, 30.0))
            
            move_backward(duration=backward_duration, velocity=0.25)
    
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

