# TIKTOK: @ericbleo

import cv2
from ultralytics import YOLO

def resize(frame, size):
    width = int(frame.shape[1] * size)
    height = int(frame.shape[0] * size)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

POSE_ESTIMATION_KEY_POINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

def classify_pose(key_points):
    
    # Acess keypoints from the pose estimation key points dictionary
    left_shoulder = key_points[5]
    right_shoulder = key_points[6]
    left_elbow = key_points[7]
    right_elbow = key_points[8]
    left_wrist = key_points[9]
    right_wrist = key_points[10]
    left_hip = key_points[11]
    right_hip = key_points[12]
    left_knee = key_points[13]
    right_knee = key_points[14]
    left_ankle = key_points[15]
    right_ankle = key_points[16]

    # IMPORTANT:
    # Keypoints are typically stored as (x, y) or (x, y, confidence). x=[0], y=[1].
    # In image coordinates, x increases to the right and y increases downward.

    # Arms raised: wrists above shoulders
    if (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] < right_shoulder[1]):
        return "Arms raised"

    # Sitting: knees close to hips vertically, ankles below knees, hips below shoulders
    # First set the deciding pixel distance between knee and hip vertically, when smaller than this value, the pose is classified as sitting.
    sitting_threshold = 40
    if (
        abs(left_knee[1] - left_hip[1]) < sitting_threshold
        and abs(right_knee[1] - right_hip[1]) < sitting_threshold
        and (left_ankle[1] > left_knee[1]) 
        and (right_ankle[1] > right_knee[1])
        and (left_hip[1] > left_shoulder[1])
        and (right_hip[1] > right_shoulder[1])
    ):
        return "Sitting"

    # Standing: hips above knees, knees above ankles, relatively upright
    if (
        left_hip[1] < left_knee[1] < left_ankle[1]
        and right_hip[1] < right_knee[1] < right_ankle[1]
    ):
        return "Standing"

    return "Unknown Pose"

def label_shape(text):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, padding = 0.5, 5
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 1)
    width = text_width + (padding*2)
    height = text_height + (padding*2)
    return width, height, text_width, text_height, font, scale, padding

def draw_centered_label(img, text, x_center, y, color):
    width, height, text_width, text_height, font, scale, padding = label_shape(text)
    x = int(x_center - width // 2)
    # Rectangle background
    cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    # Text centered inside the rectangle
    text_x = int(x + (width - text_width) // 2)
    text_y = int(y + height - padding - 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return height  # Height 4 stacking the labels

image_path = "images/picture.jpg"
model_path = "models/yolo26l-pose.pt"

pic = cv2.imread(image_path)
if pic is None:
    raise ValueError("Image not found! Check your path.")

pic = resize(pic, 1)
model = YOLO(model_path)
result = model(pic)[0]

# Not drawing model labels, only keypoints and boxes
annotated = result.plot(labels=False, conf=False, boxes=True)

for i, key_pts in enumerate(result.keypoints.xy):
    key_pts = key_pts.cpu().numpy()
    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
    x_center = int((x1 + x2) / 2)

    cls = int(result.boxes.cls[i])
    conf = float(result.boxes.conf[i])
    detection_label = f"{result.names[cls]} {int(conf*100)}%"
    pose_label = classify_pose(key_pts)
    print(f"Person {i}: {detection_label} | {pose_label}")

    # Compute BOTH label heights for stacking
    _, detection_height, _, _, _, _, _ = label_shape(detection_label)
    _, pose_height, _, _, _, _, _ = label_shape(pose_label)

    y_label_top = max(0, y1 - pose_height - detection_height - 2)
    used_pose_height = draw_centered_label(annotated, pose_label, x_center, y_label_top, (0, 200, 255))
    used_detection_height = draw_centered_label(annotated, detection_label, x_center, y_label_top + used_pose_height, (255, 128, 0))

cv2.imwrite("output/output_picture.jpg", annotated)
print("Saved to output/output_picture.jpg")

cv2.imshow("POSE ESTIMATION. TIKTOK: @ericbleo", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()