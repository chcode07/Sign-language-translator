import mediapipe as mp
import cv2 as cv
import time

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


#  GLOBAL RESULTS  

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


# REUSABLE DRAW FUNCTION 

def draw_landmarks(frame, hand_result, face_result, pose_result):
    h, w, _ = frame.shape

    # HANDS 
    if hand_result and hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for s, e in HAND_CONNECTIONS:
                cv.line(frame, pts[s], pts[e], (255, 0, 0), 2)

            for p in pts:
                cv.circle(frame, p, 4, (0, 255, 0), -1)

    #  FACE (MOUTH ONLY) 
    if face_result and face_result.face_landmarks:
        for face in face_result.face_landmarks:
            for idx in FACE_MOUTH_LANDMARKS:
                lm = face[idx]
                cv.circle(
                    frame,
                    (int(lm.x * w), int(lm.y * h)),
                    4,
                    (0, 255, 255),
                    -1
                )

    # POSE 
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
                cv.putText(
                    frame,
                    str(idx),
                    (pt[0] + 5, pt[1] - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

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

with HandLandmarker.create_from_options(hand_options) as hand_lm, \
     FaceLandmarker.create_from_options(face_options) as face_lm, \
     PoseLandmarker.create_from_options(pose_options) as pose_lm:

    cap = cv.VideoCapture(0)
    prev_time = time.time()

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

        # DRAWING USING REUSABLE METHOD 
        frame = draw_landmarks(
            frame,
            latest_hand_result,
            latest_face_result,
            latest_pose_result
        )

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(1e-6, curr_time - prev_time)
        prev_time = curr_time
        cv.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.imshow("Sign Language Tracking", frame)

        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()
