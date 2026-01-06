import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import tensorflow as tf
from collections import deque

# SETUP


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


# MODEL PATHS

hand_model_path = r"C:/Users/chann/major_project/model_task_files/hand_landmarker.task"
face_model_path = r"C:/Users/chann/major_project/model_task_files/face_landmarker.task"
pose_model_path = r"C:/Users/chann/major_project/model_task_files/pose_landmarker_heavy.task"


# GRU MODEL PATHS

gru_model_path = "signlanguage_model.h5"
label_map_path = "label_map.npy"


# LOAD GRU MODEL

model = tf.keras.models.load_model(gru_model_path)
label_map = np.load(label_map_path, allow_pickle=True).item()


# SEQUENCE CONFIG

SEQ_LEN = 90
FEATURE_DIM = 159
sequence_buffer = deque(maxlen=SEQ_LEN)


# LANDMARK DEFINITIONS

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
]

POSE_LANDMARK_IDS = [11, 12, 13, 14, 15, 16]

POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12)
]

FACE_MOUTH_LANDMARKS = [0, 13, 14, 78, 308]


# GLOBAL RESULTS (ASYNC CALLBACK OUTPUTS)

latest_hand_result = None
latest_face_result = None
latest_pose_result = None


# CALLBACKS

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result


def face_callback(result, output_image, timestamp_ms):
    global latest_face_result
    latest_face_result = result


def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result


# NORMALIZATION FUNCTIONS

def normalize_hand(hand):
    wrist = hand[0]
    out = []
    for lm in hand:
        out.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return out


def normalize_pose(pose):
    ls, rs = pose[11], pose[12]
    cx = (ls.x + rs.x) / 2
    cy = (ls.y + rs.y) / 2
    cz = (ls.z + rs.z) / 2

    out = []
    for i in POSE_LANDMARK_IDS:
        lm = pose[i]
        out.extend([
            lm.x - cx,
            lm.y - cy,
            lm.z - cz
        ])
    return out


def normalize_face(face):
    lm78, lm308 = face[78], face[308]
    cx = (lm78.x + lm308.x) / 2
    cy = (lm78.y + lm308.y) / 2
    cz = (lm78.z + lm308.z) / 2

    out = []
    for i in FACE_MOUTH_LANDMARKS:
        lm = face[i]
        out.extend([
            lm.x - cx,
            lm.y - cy,
            lm.z - cz
        ])
    return out


# REUSABLE DRAW FUNCTION

def draw_landmarks(frame, hand_result, face_result, pose_result):
    h, w, _ = frame.shape

    if hand_result and hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for s, e in HAND_CONNECTIONS:
                cv.line(frame, pts[s], pts[e], (255, 0, 0), 2)
            for p in pts:
                cv.circle(frame, p, 4, (0, 255, 0), -1)

    if face_result and face_result.face_landmarks:
        for face in face_result.face_landmarks:
            for idx in FACE_MOUTH_LANDMARKS:
                lm = face[idx]
                cv.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 255), -1)

    if pose_result and pose_result.pose_landmarks:
        for pose in pose_result.pose_landmarks:
            pts = {}
            for i in POSE_LANDMARK_IDS:
                lm = pose[i]
                pts[i] = (int(lm.x * w), int(lm.y * h))
            for s, e in POSE_CONNECTIONS:
                cv.line(frame, pts[s], pts[e], (0, 255, 255), 3)
            for idx, pt in pts.items():
                cv.circle(frame, pt, 7, (0, 0, 255), -1)

    return frame


# OPTIONS

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_callback
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=face_callback
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pose_callback
)


# RUNTIME

with ( HandLandmarker.create_from_options(hand_options) as hand_lm,
     FaceLandmarker.create_from_options(face_options) as face_lm,
     PoseLandmarker.create_from_options(pose_options) as pose_lm):

    cap = cv.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)

        hand_lm.detect_async(mp_image, timestamp)
        face_lm.detect_async(mp_image, timestamp)
        pose_lm.detect_async(mp_image, timestamp)

        frame_features = []

        if latest_hand_result and latest_hand_result.hand_landmarks:
            for hand in latest_hand_result.hand_landmarks[:2]:
                frame_features.extend(normalize_hand(hand))
        while len(frame_features) < 2 * 21 * 3:
            frame_features.extend([0.0] * (21 * 3))

        if latest_pose_result and latest_pose_result.pose_landmarks:
            frame_features.extend(normalize_pose(latest_pose_result.pose_landmarks[0]))
        else:
            frame_features.extend([0.0] * (6 * 3))

        if latest_face_result and latest_face_result.face_landmarks:
            frame_features.extend(normalize_face(latest_face_result.face_landmarks[0]))
        else:
            frame_features.extend([0.0] * (5 * 3))

        normalised_features = np.array(frame_features, dtype=np.float32)

        sequence_buffer.append(normalised_features)

        if len(sequence_buffer) == SEQ_LEN:
            sequence = np.expand_dims(np.array(sequence_buffer), axis=0)
            prediction = model.predict(sequence, verbose=0)
            class_id = np.argmax(prediction)
            class_name = label_map[class_id]
            confidence = prediction[0][class_id]
            if confidence > .8:
                print(f"Recognized: {class_name} ({confidence:.2f})")

            cv.putText(
                frame,
                f"{class_name} ({confidence:.2f})",
                (30, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        frame = draw_landmarks(frame, latest_hand_result, latest_face_result, latest_pose_result)
        cv.imshow("Sign Language Recognition", frame)

        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()
