import mediapipe as mp
import cv2 as cv
import numpy as np
import os

# CONFIG
RAW_VIDEO_DIR = r"C:/Users/chann/Downloads/new_videos"
OUTPUT_DIR = "data_book_1"

# NORMALIZATION
def normalize_and_scale(landmarks):
    if not landmarks: return [0.0] * 63
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    center = coords[0]
    coords -= center
    max_dist = np.max(np.abs(coords))
    if max_dist > 0: coords /= max_dist
    return coords.flatten().tolist()

# OPTIONS SETUP (Define options once, create instances later)
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# We create the OPTIONS here, but not the DETECTORS yet
hand_opt = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO, num_hands=2)
pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO)
face_opt = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO)

# --- MAIN LOOP ---
for class_folder in os.listdir(RAW_VIDEO_DIR):
    class_path = os.path.join(RAW_VIDEO_DIR, class_folder)
    if not os.path.isdir(class_path): continue
        
    output_class_dir = os.path.join(OUTPUT_DIR, class_folder)
    os.makedirs(output_class_dir, exist_ok=True)
    print(f"\n📂 Class: {class_folder}")

    for video_file in os.listdir(class_path):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')): continue

        print(f"  -> Processing {video_file}...", end="")
        
        # --- CRITICAL FIX: RE-INITIALIZE MEDIAPIPE FOR EVERY VIDEO ---
        # This resets the internal timestamp memory so we can start from 0 again.
        with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
             mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
             mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:

            cap = cv.VideoCapture(os.path.join(class_path, video_file))
            fps = cap.get(cv.CAP_PROP_FPS)
            if fps == 0: fps = 30
            
            sequence = []
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Timestamp Calculation
                ts = int((frame_idx / fps) * 1000)
                
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                
                # Detect
                hand_res = hand_lm.detect_for_video(mp_img, ts)
                pose_res = pose_lm.detect_for_video(mp_img, ts)
                face_res = face_lm.detect_for_video(mp_img, ts)
                
                feats = []

                # --- HAND SORTING (Right vs Left) ---
                right_hand, left_hand = None, None
                if hand_res.handedness:
                    for idx, hand_meta in enumerate(hand_res.handedness):
                        label = hand_meta[0].category_name 
                        landmarks = hand_res.hand_landmarks[idx]
                        if label == "Right": right_hand = landmarks
                        elif label == "Left": left_hand = landmarks

                feats.extend(normalize_and_scale(right_hand)) # Right (0-63)
                feats.extend(normalize_and_scale(left_hand))  # Left (64-126)
                
                # Pose
                if pose_res.pose_landmarks:
                    subset = [pose_res.pose_landmarks[0][i] for i in [11,12,13,14,15,16]]
                    feats.extend(normalize_and_scale(subset))
                else: feats.extend([0.0] * 18)
                    
                # Face
                if face_res.face_landmarks:
                    subset = [face_res.face_landmarks[0][i] for i in [0, 13, 14, 78, 308]]
                    feats.extend(normalize_and_scale(subset))
                else: feats.extend([0.0] * 15)
                
                sequence.append(feats)
                frame_idx += 1

            cap.release()
            
            # Save
            save_name = os.path.splitext(video_file)[0] + ".npy"
            np.save(os.path.join(output_class_dir, save_name), np.array(sequence, dtype=np.float32))
            print(" Done.")

print("\n✅ Extraction Complete.")


