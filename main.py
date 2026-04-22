from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Run detection
results = model("test.jpg")

# Get detections
boxes = results[0].boxes

detections = []

for box in boxes:
    cls = int(box.cls[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    detections.append({
        "class": cls,
        "box": [x1, y1, x2, y2]
    })

# IOU function
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1)*max(0, y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter/union if union else 0

# Pickability scoring
for i in range(len(detections)):
    x1, y1, x2, y2 = detections[i]["box"]
    area = (x2-x1)*(y2-y1)

    overlap = 0
    for j in range(len(detections)):
        if i != j:
            overlap += compute_iou(
                detections[i]["box"],
                detections[j]["box"]
            )

    detections[i]["score"] = 0.6*area - 0.4*overlap

# Rank objects
detections.sort(key=lambda x: x["score"], reverse=True)

# Show results
img = cv2.imread("test.jpg")

for i, obj in enumerate(detections):
    x1, y1, x2, y2 = map(int, obj["box"])
    score = obj["score"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"{i+1}: {score:.2f}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()