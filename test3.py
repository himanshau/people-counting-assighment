
# IMPORT REQUIRED LIBRARIES
from ultralytics import YOLO  # YOLOv8 model
import cv2  # For video processing
import cvzone  # For display and drawing
import numpy as np  # For matrix operations
from sort import *  # SORT tracker for multi-object tracking

# LOAD YOLO MODEL
model = YOLO("yolov8m.pt")  # Load pre-trained YOLOv8 model
model.to("cuda")  # Use GPU for faster inference

#  LOAD INPUT VIDEO AND MASK
cap = cv2.VideoCapture("vidp.mp4")  # Input video
mask = cv2.imread("r.png")  # Load region mask image

# Resize mask if it doesn't match the video frame size
if mask is not None:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = cv2.resize(mask, (frame_width, frame_height))

# SETUP OUTPUT VIDEO WRITER
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
)

#  INITIALIZE TRACKER AND VARIABLES
# Optimized SORT tracker parameters:
# - max_age: frames to keep alive a track without detections (increased for occlusions)
# - min_hits: minimum detections before track is confirmed (reduced for faster response)
# - iou_threshold: intersection over union threshold for matching (increased for stability)
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
limits = [550, 550, 1330, 550]  # Line coordinates for counting (y = 550)

# Class names for YOLO COCO dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
              "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Store object movement history and crossing states
track_history = {}
count_up = set()
count_down = set()
crossed_objects = {}  # Track which side of line each object is on

# 🔁 PROCESS VIDEO FRAMES
frame_count = 0
while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    if frame_count % 30 == 0:  # Progress update every 30 frames
        print(f"Frame {frame_count} | UP: {len(count_up)} | DOWN: {len(count_down)}")

    # Apply mask to focus detection only on allowed region
    if mask is not None:
        videoregion = cv2.bitwise_and(img, mask)
    else:
        videoregion = img

    # Run YOLO detection with optimized parameters
    results = model(videoregion, stream=True, device=0, conf=0.3, iou=0.5)

    # Collect detections in [x1, y1, x2, y2, conf] format
    detections = np.empty((0, 5))
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Detect only persons with confidence > 0.3 (lowered for better detection)
            if classNames[cls] == "person" and conf > 0.3:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    # Update tracker with detections
    results_tracker = tracker.update(detections)

    # Draw the counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

    # Process tracked objects
    for result in results_tracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        # Draw box and center
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=8, t=2, colorR=(255, 0, 0))
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.putTextRect(img, f'ID {id}', (x1, max(35, y1 - 20)), 1, 1)

        # Initialize tracking for new objects
        if id not in track_history:
            track_history[id] = []
            crossed_objects[id] = None  # None = not yet determined which side
        
        # Save position history
        track_history[id].append(cy)

        # Keep last 10 positions for better direction detection
        if len(track_history[id]) > 10:
            track_history[id] = track_history[id][-10:]

        # Line crossing detection - only count when object crosses the line
        line_y = limits[1]
        
        # Check if object has enough history to determine crossing
        if len(track_history[id]) >= 3:
            prev_y = track_history[id][-2]  # Previous position
            curr_y = cy  # Current position
            
            # Determine which side of the line the object is on
            if crossed_objects[id] is None:
                # First time seeing this object, record its side
                if curr_y < line_y:
                    crossed_objects[id] = "above"
                else:
                    crossed_objects[id] = "below"
            else:
                # Check if object crossed the line
                # Moving DOWN: was above line, now below
                if crossed_objects[id] == "above" and prev_y < line_y and curr_y >= line_y:
                    if id not in count_down:
                        count_down.add(id)
                        crossed_objects[id] = "below"
                        # Draw visual feedback
                        cv2.circle(img, (cx, line_y), 15, (0, 0, 255), cv2.FILLED)
                
                # Moving UP: was below line, now above
                elif crossed_objects[id] == "below" and prev_y > line_y and curr_y <= line_y:
                    if id not in count_up:
                        count_up.add(id)
                        crossed_objects[id] = "above"
                        # Draw visual feedback
                        cv2.circle(img, (cx, line_y), 15, (0, 255, 0), cv2.FILLED)

    # Display counts on screen
    cvzone.putTextRect(img, f'UP: {len(count_up)}', (50, 50), 2, 2, colorT=(0, 0, 0), colorR=(0, 255, 0))
    cvzone.putTextRect(img, f'DOWN: {len(count_down)}', (50, 110), 2, 2, colorT=(0, 0, 0), colorR=(0, 0, 255))

    # Save frame to output video
    out.write(img)

    # Display live window
    cv2.imshow("YOLO Person Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  CLEANUP
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing completed successfully! Output saved as 'output.mp4'")
