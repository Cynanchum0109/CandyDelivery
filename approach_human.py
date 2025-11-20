import cv2
from ultralytics import YOLO
import subprocess

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
# YOLO HUMAN DETECTION SCRIPT
# ===========================================
model = YOLO("yolov8n.pt")      # YOLO model

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

TARGET_HEIGHT = 280             # stopping threshold

print("Human follower started (YOLO + SSH)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame, verbose=False)
    max_height = 0

    # Parse YOLO results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                if h > max_height:
                    max_height = h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Movement logic
    if max_height == 0:
        print("No human → STOP")
        send_vel(0.0)

    else:
        print(f"Human detected, height = {max_height}")

        if max_height < TARGET_HEIGHT:
            print("Too far → MOVE FORWARD")
            send_vel(0.20)
        else:
            print("Close enough → STOP")
            send_vel(0.0)

    cv2.imshow("YOLO Human Detection (Laptop)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
send_vel(0.0)   # stop robot on exit
