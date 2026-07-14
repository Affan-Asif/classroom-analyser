import cv2
from ultralytics import YOLO

# Load local YOLOv11 model (same folder OR give full path)
model = YOLO(r"C:\Affan\Affan\__CODING__\__PROJECTS__\Classroom Analyser\yolo11m.pt")

# Use webcam (0) OR video file
cap = cv2.VideoCapture(0)   # or "class.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Draw results
    frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv11 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
