import mediapipe as mp
import cv2 as cv
import numpy as np
import os

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

POSE_LANDMARK_IDS = [11, 12, 13, 14, 15, 16]
FACE_MOUTH_LANDMARKS = [0, 13, 14, 78, 308]

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
]

POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12)
]


# NORMALIZATION 

def normalize_hand(hand):
    wrist = hand[0]
    out = []
    for lm in hand:
        out.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return out


def normalize_pose(pose):
    ls, rs = pose[11], pose[12]
    cx = (ls.x + rs.x) / 2
    cy = (ls.y + rs.y) / 2
    cz = (ls.z + rs.z) / 2

    out = []
    for i in POSE_LANDMARK_IDS:
        lm = pose[i]
        out.extend([lm.x - cx, lm.y - cy, lm.z - cz])
    return out


def normalize_face(face):
    lm78, lm308 = face[78], face[308]
    cx = (lm78.x + lm308.x) / 2
    cy = (lm78.y + lm308.y) / 2
    cz = (lm78.z + lm308.z) / 2

    out = []
    for i in FACE_MOUTH_LANDMARKS:
        lm = face[i]
        out.extend([lm.x - cx, lm.y - cy, lm.z - cz])
    return out


# DRAW FUNCTION (REUSABLE) 

def draw_landmarks(frame, hand_result, face_result, pose_result):
    h, w, _ = frame.shape

    # Hands
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for s, e in HAND_CONNECTIONS:
                cv.line(frame, pts[s], pts[e], (255, 0, 0), 2)
            for p in pts:
                cv.circle(frame, p, 4, (0, 255, 0), -1)

    # Face (mouth only)
    if face_result.face_landmarks:
        for face in face_result.face_landmarks:
            for idx in FACE_MOUTH_LANDMARKS:
                lm = face[idx]
                cv.circle(frame, (int(lm.x * w), int(lm.y * h)), 4,
                          (0, 255, 255), -1)

    # Pose
    if pose_result.pose_landmarks:
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
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO
)


# RUNTIME 
for file_name in os.listdir("C:/Users/chann/Downloads/training_videos"):
        
        video_path = f"C:/Users/chann/Downloads/training_videos/{file_name}"
        name_without_ext, _ = os.path.splitext(file_name)
        new_dir_path = os.path.join("C:/Users/chann/major_project/data_book/", name_without_ext)

        try:
            os.makedirs(new_dir_path, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory for {name_without_ext}: {e}")


        output_npy = f"C:/Users/chann/major_project/data_book/{name_without_ext}/video_0.npy"

        sequence = []

        with (HandLandmarker.create_from_options(hand_options) as hand_lm,
            FaceLandmarker.create_from_options(face_options) as face_lm,
            PoseLandmarker.create_from_options(pose_options) as pose_lm):

            cap = cv.VideoCapture(video_path)

            fps = cap.get(cv.CAP_PROP_FPS)
            if fps == 0:
                fps = 30

            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(frame_idx * 1000 / fps)

                hand_result = hand_lm.detect_for_video(mp_image, timestamp)
                face_result = face_lm.detect_for_video(mp_image, timestamp)
                pose_result = pose_lm.detect_for_video(mp_image, timestamp)

                # FEATURE EXTRACTION 
                frame_features = []

                if hand_result.hand_landmarks:
                    for hand in hand_result.hand_landmarks[:2]:
                        frame_features.extend(normalize_hand(hand))
                while len(frame_features) < 2 * 21 * 3:
                    frame_features.extend([0.0] * (21 * 3))

                if pose_result.pose_landmarks:
                    frame_features.extend(normalize_pose(pose_result.pose_landmarks[0]))
                else:
                    frame_features.extend([0.0] * (6 * 3))

                if face_result.face_landmarks:
                    frame_features.extend(normalize_face(face_result.face_landmarks[0]))
                else:
                    frame_features.extend([0.0] * (5 * 3))

                sequence.append(frame_features)

                # # VISUALIZATION (only for verification)
                # frame = draw_landmarks(frame, hand_result, face_result, pose_result)
                # cv.imshow("Landmark Visualization", frame)

                # if cv.waitKey(1) & 0xFF in (27, ord('q')):
                #     break

                frame_idx += 1

            cap.release()
            cv.destroyAllWindows()


        # SAVE 

        sequence = np.array(sequence, dtype=np.float32)
        print("Final sequence shape:", sequence.shape)
        np.save(output_npy, sequence)
        print(f"Saved to {output_npy}")
