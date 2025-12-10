import cv2
from ultralytics import YOLO
import subprocess
import threading
import queue
import time
import numpy as np
from datetime import datetime

# ========= CONFIGURE SSH INFO HERE =========
ROBOT_USER = "turtle-one"                  # your robot's username
ROBOT_IP = "10.5.12.184"             # replace with robot IP
SSH = f"{ROBOT_USER}@{ROBOT_IP}"
# ===========================================

def ssh_cmd(command):
    """Send a command to the robot over SSH."""
    full_cmd = f'ssh {SSH} "{command}"'
    subprocess.run(full_cmd, shell=True)

def send_vel(linear_x, angular_z=0.0):
    # Build YAML message
    yaml_msg = (
        "{linear: {x: " + str(linear_x) + ", y: 0.0, z: 0.0}, "
        "angular: {x: 0.0, y: 0.0, z: " + str(angular_z) + "}}"
    )

    # Correct quoting: single quotes around everything, YAML inside double quotes
    cmd = (
        f"ssh {SSH} "
        f"\"rostopic pub -1 /cmd_vel_mux/input/navi geometry_msgs/Twist "
        f"\\\"{yaml_msg}\\\"\""
    )

    subprocess.run(cmd, shell=True)


# ===========================================
# CONTINUOUS STREAMING YOLO HUMAN DETECTION
# ===========================================

# Configuration
TARGET_HEIGHT = 280             # stopping threshold
FPS_TARGET = 30                # target frames per second
CAMERA_ID = 0                  # camera device ID

# Initialize YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully.")

# Initialize camera with optimized settings for streaming
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

# Optimize camera settings for continuous streaming
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer to reduce latency

# Get actual frame dimensions for video recording
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"human_detection_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
video_writer = cv2.VideoWriter(video_filename, fourcc, FPS_TARGET, (frame_width, frame_height))

if not video_writer.isOpened():
    print(f"Warning: Could not initialize video writer for {video_filename}")
    video_writer = None
else:
    print(f"Recording video to: {video_filename}")

# Frame queue for multi-threaded processing
frame_queue = queue.Queue(maxsize=2)     # Small buffer to avoid lag
result_queue = queue.Queue(maxsize=2)

# Control flags
running = True
last_vel_command = None
vel_lock = threading.Lock()

def capture_frames():
    """Continuously capture frames from camera in a separate thread."""
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            time.sleep(0.1)
            continue
        
        timestamp = time.time()
        
        # Write raw frame directly to video (original video stream)
        if video_writer:
            video_writer.write(frame)
        
        # Put frame in detection queue (drop old frames if queue is full)
        if not frame_queue.full():
            frame_queue.put((timestamp, frame))
        else:
            # Remove old frame and add new one
            try:
                frame_queue.get_nowait()
                frame_queue.put((timestamp, frame))
            except queue.Empty:
                pass

def process_detections():
    """Process frames with YOLO in a separate thread."""
    global running
    while running:
        try:
            # Get frame from queue with timeout
            timestamp, frame = frame_queue.get(timeout=0.1)
            
            # Run YOLO detection
            results = model(frame, verbose=False, imgsz=640)  # Fixed size for speed
            
            # Parse results
            max_height = 0
            annotated_frame = frame.copy()
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h = y2 - y1
                        if h > max_height:
                            max_height = h
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Person: {h}px", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Put result in queue
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

print("Human follower started (Continuous Streaming YOLO + SSH)...")
print("Press 'q' to quit")

# Start capture and processing threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_detections, daemon=True)

capture_thread.start()
process_thread.start()

# Main loop: display results and control robot
frame_time = 1.0 / FPS_TARGET
last_display_time = time.time()

try:
    while True:
        current_time = time.time()
        
        # Control display frame rate
        if current_time - last_display_time < frame_time:
            time.sleep(0.01)  # Small sleep to avoid busy waiting
            continue
        
        last_display_time = current_time
        
        try:
            # Get latest result
            timestamp, annotated_frame, max_height = result_queue.get(timeout=0.1)
            
            # Calculate FPS
            fps = 1.0 / (current_time - timestamp) if current_time > timestamp else 0
            
            # Add FPS display
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Movement logic
            if max_height == 0:
                status_text = "No human → STOP"
                send_vel(0.0)
            else:
                if max_height < TARGET_HEIGHT:
                    status_text = f"Human detected (h={max_height}) → MOVE FORWARD"
                    send_vel(0.20)
                else:
                    status_text = f"Human detected (h={max_height}) → CLOSE ENOUGH → STOP"
                    send_vel(0.0)
            
            # Add status text
            cv2.putText(annotated_frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add recording indicator
            if video_writer:
                cv2.putText(annotated_frame, "REC", (frame_width - 80, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (frame_width - 30, 30), 8, (0, 0, 255), -1)
            
            # Note: Raw frames are written directly in capture_frames() thread
            # This ensures we record the original video stream, not annotated frames
            
            # Display annotated frame
            cv2.imshow("YOLO Human Detection (Continuous Streaming)", annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except queue.Empty:
            # No results yet, show waiting message
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Waiting for detection...", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("YOLO Human Detection (Continuous Streaming)", blank_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    running = False
    time.sleep(0.5)  # Give threads time to finish
    
    # Release video writer
    if video_writer:
        video_writer.release()
        print(f"Video saved to: {video_filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    send_vel(0.0)   # stop robot on exit
    print("Streaming stopped. Robot halted.")
