import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe for hands, face, and full-body pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face = mp_face.FaceMesh(min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load YOLOv8 for human detection
model = YOLO('yolov8n.pt')  # You can switch to 'yolov8s.pt' for better accuracy

# Webcam capture
cap = cv2.VideoCapture("cv_sample.mp4")
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Tracking
    hand_results = hands.process(imgRGB)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Face Tracking
    face_results = face.process(imgRGB)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

    # Full Body Pose Tracking
    pose_results = pose.process(imgRGB)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # YOLO Human Detection
    yolo_results = model(frame)
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("AR + Hand + Face + Pose + Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
