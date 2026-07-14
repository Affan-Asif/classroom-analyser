# from ultralytics import YOLO
# import cv2
# # Load the pre-trained YOLO model
# model = YOLO("best.pt")
# class_names = [
#     "Affan",        # Class 0
#     "Using_mobile", # Class 1
#     "Faizan",       # Class 2
#     "working",      # Class 3
#     "gossiping",    # Class 4
#     "sleeping",     # Class 5
#     "shafaq"        # Class 6
# ]
# # Perform prediction using the model
# # cap = cv2.VideoCapture("class_resized.mp4") #input video name
# model.predict(
#     source="class_resized.mp4",      # Input video file
#     show=True,               # Show the predictions in a visualization window
#     save=True,               # Save the results (annotated images or videos)
#     conf=0.7,                # Confidence threshold for predictions
#     line_width=1,            # Line width for bounding boxes
#     save_crop=False,         # Whether to save cropped images of detected objects
#     save_txt=False,          # Whether to save results in a text file
#     show_labels=True,        # Show class labels on bounding boxes
#     show_conf=True,          # Show confidence scores on bounding boxes
#     classes=list(range(7))   # Detect all 7 classes (0 to 6)
# )

import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("best.pt")

# Define class names
class_names = [
    "Affan",        # Class 0
    "Using_mobile", # Class 1
    "Faizan",       # Class 2
    "working",      # Class 3
    "gossiping",    # Class 4
    "sleeping",     # Class 5
    "shafaq"        # Class 6
]

# Initialize the webcam feed (use 0 for default webcam, or the appropriate index for external webcams)
camera_index = "class.mp4"
cap = cv2.VideoCapture(camera_index)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam {camera_index}")
    exit()

# Loop to capture frames from the webcam and perform detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If the frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform detection using the model
    results = model.predict(
        source=frame,
        conf=0.7,                # Confidence threshold for predictions
        line_width=1,            # Line width for bounding boxes
        save_crop=False,         # Whether to save cropped images of detected objects
        save_txt=False,          # Whether to save results in a text file
        show_labels=True,        # Show class labels on bounding boxes
        show_conf=True,          # Show confidence scores on bounding boxes
        classes=list(range(7))   # Detect all 7 classes (0 to 6)
    )

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
