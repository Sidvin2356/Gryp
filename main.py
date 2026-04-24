from ultralytics import YOLO
import cv2

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = "test.jpg"
GRIPPER = "parallel"   # options: "parallel", "suction", "3finger"

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# RUN DETECTION
# -------------------------------
results = model(IMAGE_PATH)
boxes = results[0].boxes
names = model.names

detections = []

for box in boxes:
    cls = int(box.cls[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    detections.append({
        "class": cls,
        "label": names[cls],
        "box": [x1, y1, x2, y2]
    })

# -------------------------------
# IOU FUNCTION
# -------------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter/union if union else 0

# -------------------------------
# SHAPE ESTIMATION
# -------------------------------
def get_shape(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    ratio = w / h if h != 0 else 1

    if ratio > 1.5:
        return "flat"
    elif ratio < 0.7:
        return "tall"
    else:
        return "regular"

# -------------------------------
# GRIPPER COMPATIBILITY
# -------------------------------
def gripper_score(shape, gripper):
    if gripper == "parallel":
        if shape == "regular":
            return 1.0
        elif shape == "tall":
            return 0.8
        else:
            return 0.5

    elif gripper == "suction":
        if shape == "flat":
            return 1.0
        else:
            return 0.4

    elif gripper == "3finger":
        if shape == "regular":
            return 0.9
        else:
            return 0.7

    return 0.5

# -------------------------------
# LOAD IMAGE
# -------------------------------
img = cv2.imread(IMAGE_PATH)
img_area = img.shape[0] * img.shape[1]

# -------------------------------
# PICKABILITY SCORING
# -------------------------------
for i in range(len(detections)):
    x1, y1, x2, y2 = detections[i]["box"]

    # SIZE (normalized)
    area = (x2 - x1) * (y2 - y1)
    norm_area = area / img_area

    # OVERLAP
    overlap = 0
    for j in range(len(detections)):
        if i != j:
            overlap += compute_iou(
                detections[i]["box"],
                detections[j]["box"]
            )

    # SHAPE
    shape = get_shape(detections[i]["box"])

    # GRIPPER COMPATIBILITY
    g_score = gripper_score(shape, GRIPPER)

    # FINAL SCORE
    detections[i]["score"] = (
        0.5 * norm_area +
        0.3 * g_score -
        0.2 * overlap
    )

    detections[i]["shape"] = shape

# -------------------------------
# SORT BY PICKABILITY
# -------------------------------
detections.sort(key=lambda x: x["score"], reverse=True)

# -------------------------------
# VISUALIZE RESULTS
# -------------------------------
for i, obj in enumerate(detections):
    x1, y1, x2, y2 = map(int, obj["box"])
    score = obj["score"]
    label = obj["label"]
    shape = obj["shape"]

    text = f"{i+1}: {label} ({shape}) {score:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, text,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)

# Show gripper type
cv2.putText(img, f"Gripper: {GRIPPER}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,255,255), 2)

cv2.imshow("Pickability Ranking", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n--- PICKABILITY RANKING ---")
for i, obj in enumerate(detections):
    print(f"{i+1}. {obj['label']} | Shape: {obj['shape']} | Score: {obj['score']:.2f}")