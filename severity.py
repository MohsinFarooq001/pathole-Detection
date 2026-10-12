import cv2
from ultralytics import YOLO

# Load pothole detection model
model = YOLO("best.pt")

# Input video
cap = cv2.VideoCapture("videos/Pathole Road.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Output video
out = cv2.VideoWriter(
    "pothole_severity_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection - CPU
    results = model(
        frame,
        conf=0.25,
        imgsz=416,
        device="cpu",
        verbose=False
    )[0]

    for box in results.boxes.xyxy:

        x1, y1, x2, y2 = map(int, box)

        # Calculate pothole area
        area = (x2 - x1) * (y2 - y1)

        # Severity classification
        if area < 3000:
            severity = "LOW"
        elif area < 10000:
            severity = "MEDIUM"
        else:
            severity = "HIGH"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw severity
        cv2.putText(
            frame,
            severity,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Save processed frame
    out.write(frame)

    # Show result
    cv2.imshow(
        "Pothole Detection & Severity",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("================================")
print("VIDEO PROCESSING COMPLETED")
print("Device: CPU")
print("YOLO Image Size: 416")
print("Saved as: pothole_severity_output.mp4")
print("================================")