import mediapipe as mp
import cv2 as cv
import time
import csv
import os

# --- Tasks API Aliases ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
FaceLandmarker = mp.tasks.vision.FaceLandmarker

# --- Configuration Paths ---
# Ensure these files are in your project folder
hand_model_path = r"C:/Users/chann/major_project/model_taskfiles/hand_landmarker.task"
face_model_path = r"C:/Users/chann/major_project/model_taskfiles/face_landmarker.task"
pose_model_path = r"C:/Users/chann/major_project/model_taskfiles/pose_landmarker_heavy.task"

# --- Global Result Buffers ---
hand_results = None
pose_results = None
face_results = None

# --- Callbacks ---
def hand_callback(result, output_image, timestamp_ms):
    global hand_results
    hand_results = result

def pose_callback(result, output_image, timestamp_ms):
    global pose_results
    pose_results = result

def face_callback(result, output_image, timestamp_ms):
    global face_results
    face_results = result

# --- Initialization Options ---
common_options = {"running_mode": VisionRunningMode.LIVE_STREAM}

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    num_hands=2, result_callback=hand_callback, **common_options)

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    result_callback=pose_callback, **common_options)

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    result_callback=face_callback, **common_options)

# --- Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# --- Runtime ---
with HandLandmarker.create_from_options(hand_options) as hand_lm, \
     PoseLandmarker.create_from_options(pose_options) as pose_lm, \
     FaceLandmarker.create_from_options(face_options) as face_lm:
    
    vid = cv.VideoCapture(0)
    prev_time = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret: break
        
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        # 1. Trigger Async Detection
        hand_lm.detect_async(mp_image, timestamp)
        pose_lm.detect_async(mp_image, timestamp)
        face_lm.detect_async(mp_image, timestamp)

        # 2. Draw Hand Landmarks
        if hand_results and hand_results.hand_landmarks:
            for landmarks in hand_results.hand_landmarks:
                hand_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                ])
                mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)

        # 3. Draw Pose Landmarks
        if pose_results and pose_results.pose_landmarks:
            for landmarks in pose_results.pose_landmarks:
                pose_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                ])
                mp_drawing.draw_landmarks(frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS)

        # 4. Draw Face Mesh
        if face_results and face_results.face_landmarks:
            for landmarks in face_results.face_landmarks:
                face_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                ])
                mp_drawing.draw_landmarks(frame, face_landmarks_proto, mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

        # Display and Exit
        cv.imshow("Holistic (New API)", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    cv.destroyAllWindows()