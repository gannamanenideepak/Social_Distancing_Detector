import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize MQTT client
broker = "mqtt.eclipseprojects.io"
topic = "social_distancing_alerts"
client = mqtt.Client()

# Initialize Excel file with more detailed columns
excel_file = "social_distancing_data.xlsx"
try:
    df = pd.read_excel(excel_file)
except:
    df = pd.DataFrame(columns=[
        'Timestamp',
        'Total People Detected',
        'Number of Violations',
        'Violation Details',
        'Average Distance (m)',
        'Min Distance (m)',
        'Max Distance (m)',
        'Frame Processing Time (ms)'
    ])
    df.to_excel(excel_file, index=False)

try:
    client.connect(broker, 1883, 60)
except:
    print("Could not connect to MQTT broker. Will continue without MQTT alerts.")

# Use IP Webcam
PHONE_IP = "192.168.0.24"  # Your phone's IP
STREAM_URL = f"http://{PHONE_IP}:8080/video"

print(f"Attempting to connect to IP Webcam at {STREAM_URL}")
print("Make sure:")
print("1. Your phone and computer are on the same WiFi network")
print("2. IP Webcam app is running and server is started")
print("3. The IP address matches what's shown in the IP Webcam app")

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Could not connect to IP Webcam.")
    print("Please check if:")
    print("1. The IP address is correct")
    print("2. IP Webcam app is running")
    print("3. Both devices are on the same network")
    exit()

# Define safe distance threshold (in meters)
SAFE_DISTANCE = 2.0  # 2 meters

# Average person height in meters (used for calibration)
AVERAGE_PERSON_HEIGHT = 1.7

# Camera calibration parameters
KNOWN_DISTANCE = 2.0  # Known distance in meters for calibration
KNOWN_HEIGHT = 1.7    # Known height in meters for calibration

# Add these variables before the main loop
last_excel_update = time.time()
EXCEL_UPDATE_INTERVAL = 10  # Update Excel every 10 seconds

def get_ground_plane_points(frame):
    """Get points for ground plane calibration"""
    height, width = frame.shape[:2]
    
    # Define 4 points on the ground plane
    bottom_left = (width * 0.1, height * 0.9)
    bottom_right = (width * 0.9, height * 0.9)
    top_left = (width * 0.1, height * 0.7)
    top_right = (width * 0.9, height * 0.7)
    
    # Define corresponding points in real-world coordinates
    real_bottom_left = (0, 0)
    real_bottom_right = (KNOWN_DISTANCE, 0)
    real_top_left = (0, KNOWN_DISTANCE)
    real_top_right = (KNOWN_DISTANCE, KNOWN_DISTANCE)
    
    return np.float32([bottom_left, bottom_right, top_right, top_left]), \
           np.float32([real_bottom_left, real_bottom_right, real_top_right, real_top_left])

def calibrate_camera(frame, known_distance, known_height):
    """Calibrate camera using ground plane and person height"""
    results = model(frame)
    max_height = 0
    calibration_factor = 0
    
    # Get ground plane points
    src_points, dst_points = get_ground_plane_points(frame)
    
    # Calculate homography matrix for ground plane
    homography_matrix = cv2.findHomography(src_points, dst_points)[0]
    
    # Find person for height calibration
    person_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                person_detected = True
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                height_pixels = y2 - y1
                if height_pixels > max_height:
                    max_height = height_pixels
                    # Calculate calibration factor using homography
                    person_bottom = np.array([[x1, y2], [x2, y2]], dtype=np.float32)
                    real_coords = cv2.perspectiveTransform(person_bottom.reshape(-1, 1, 2), homography_matrix)
                    real_width = np.linalg.norm(real_coords[1] - real_coords[0])
                    calibration_factor = known_height / height_pixels
    
    if not person_detected:
        print("No person detected in frame. Please make sure you are visible in the camera.")
        return 0, None
    
    return calibration_factor, homography_matrix

def calculate_distance(p1, p2, homography_matrix):
    """Calculate real-world distance using homography"""
    # Convert points to numpy array
    points = np.array([[p1], [p2]], dtype=np.float32)
    
    # Transform points to real-world coordinates
    real_coords = cv2.perspectiveTransform(points, homography_matrix)
    
    # Calculate distance in meters
    distance = np.linalg.norm(real_coords[1] - real_coords[0])
    
    return distance

def draw_warning(frame, text, position, color=(0, 0, 255)):
    """Draw warning text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    padding = 10
    cv2.rectangle(frame, 
                 (position[0] - padding, position[1] - text_height - padding),
                 (position[0] + text_width + padding, position[1] + padding),
                 (255, 255, 255), -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

# Calibration phase
print("Calibrating camera...")
print("Please stand at a known distance (2 meters) from the camera")
print("Make sure you are fully visible in the frame")
print("Press 'c' when ready to calibrate, or 'q' to quit")

calibration_factor = 0
homography_matrix = None
calibration_attempts = 0
max_attempts = 5

while calibration_factor == 0 and calibration_attempts < max_attempts:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame during calibration")
        break
    
    # Draw calibration instructions
    cv2.putText(frame, "Press 'c' to calibrate", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Make sure you are visible", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Attempt {calibration_attempts + 1}/{max_attempts}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show detection boxes for feedback
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imshow("Calibration", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        calibration_attempts += 1
        print(f"\nCalibration attempt {calibration_attempts}/{max_attempts}")
        calibration_factor, homography_matrix = calibrate_camera(frame, KNOWN_DISTANCE, KNOWN_HEIGHT)
        if calibration_factor > 0:
            print(f"Calibration complete! Factor: {calibration_factor:.4f}")
        else:
            print("Calibration failed. Please try again.")
    elif key == ord('q'):
        print("Calibration cancelled")
        break

if calibration_factor == 0:
    print("Failed to calibrate camera after", max_attempts, "attempts.")
    print("Please check if:")
    print("1. You are visible in the camera")
    print("2. The lighting is good")
    print("3. You are standing at the correct distance (2 meters)")
    cap.release()
    cv2.destroyAllWindows()
    exit()

print("Starting social distancing monitoring...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from IP Webcam")
        break
    
    # Start timing frame processing
    start_time = time.time()
    
    # Detect objects in the frame
    results = model(frame)
    
    people = []
    distances = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0:  # Person class
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                people.append((int(center_x), int(center_y)))
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Check distance between detected people
    violation_detected = False
    violation_count = 0
    violation_details = []
    
    for i, p1 in enumerate(people):
        for j, p2 in enumerate(people):
            if i != j:
                # Calculate real-world distance using homography
                distance_meters = calculate_distance(p1, p2, homography_matrix)
                distances.append(distance_meters)
                
                # Draw line between people with color based on distance
                if distance_meters < SAFE_DISTANCE:
                    color = (0, 0, 255)  # Red for violation
                    violation_detected = True
                    violation_count += 1
                    violation_details.append(f"Pair {i+1}-{j+1}: {distance_meters:.2f}m")
                    # Draw warning circle
                    cv2.circle(frame, p1, 10, (0, 0, 255), -1)
                    cv2.circle(frame, p2, 10, (0, 0, 255), -1)
                else:
                    color = (0, 255, 0)  # Green for safe
                    # Only show distance on green lines
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    cv2.putText(frame, f"{distance_meters:.1f}m", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.line(frame, p1, p2, color, 2)
                
                if distance_meters < SAFE_DISTANCE:
                    # Draw warning text
                    warning_text = "MAINTAIN SOCIAL DISTANCE!"
                    draw_warning(frame, warning_text, (10, 30))

    # Calculate frame processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Display overall status
    status_text = f"Violations: {violation_count}"
    status_color = (0, 0, 255) if violation_detected else (0, 255, 0)
    draw_warning(frame, status_text, (10, 60), status_color)

    # Update Excel every 10 seconds
    current_time = time.time()
    if current_time - last_excel_update >= EXCEL_UPDATE_INTERVAL:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_distance = np.mean(distances) if distances else 0
            min_distance = np.min(distances) if distances else 0
            max_distance = np.max(distances) if distances else 0
            
            new_row = pd.DataFrame({
                'Timestamp': [timestamp],
                'Total People Detected': [len(people)],
                'Number of Violations': [violation_count],
                'Violation Details': ['; '.join(violation_details)],
                'Average Distance (m)': [f"{avg_distance:.2f}"],
                'Min Distance (m)': [f"{min_distance:.2f}"],
                'Max Distance (m)': [f"{max_distance:.2f}"],
                'Frame Processing Time (ms)': [f"{processing_time:.2f}"]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(excel_file, index=False)
            last_excel_update = current_time
            
            # Send MQTT alert if violations detected
            if violation_detected:
                client.publish(topic, f"⚠ Social Distancing Violation Detected! {violation_count} pairs too close.")
                
        except Exception as e:
            print(f"Error updating Excel or sending MQTT alert: {e}")

    # Show the frame
    cv2.imshow("Social Distancing Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
try:
    client.disconnect()
except:
    pass