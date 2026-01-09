import mediapipe as mp
import cv2 as cv
import numpy as np
import tensorflow as tf
from collections import deque
import time

#  1. CONFIGURATION 
# Loading model files
model = tf.keras.models.load_model("./signlanguage_model.keras", compile=False)
label_map = np.load("./label_map.npy", allow_pickle=True).item()

SEQ_LEN = 90  # matching TARGET_LENGTH in augmentation
PREDICTION_FREQ = 5 # Run prediction every 5 frames to prevent lag
confidence_threshold = 0.75 # Only show prediction if confident

# Buffer to store the sliding window of frames
sequence_buffer = deque(maxlen=SEQ_LEN)
frame_cnt = 0

#  2. CRITICAL: ALIGNED NORMALIZATION 
def normalize_and_scale(landmarks):
   # Must be IDENTICAL to extract_data.py
    if not landmarks: return [0.0] * 63
    
    # Convert to numpy
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Centering (Relative to first point)
    center = coords[0]
    coords -= center
    
    # Scaling (0 to 1 range)
    max_dist = np.max(np.abs(coords))
    if max_dist > 0: 
        coords /= max_dist
        
    return coords.flatten().tolist()

#  3. MEDIAPIPE CALLBACKS 
latest_hands = None
latest_pose = None
latest_face = None

def h_cb(result, output_image, timestamp_ms):
    global latest_hands
    latest_hands = result

def p_cb(result, output_image, timestamp_ms):
    global latest_pose
    latest_pose = result

def f_cb(result, output_image, timestamp_ms):
    global latest_face
    latest_face = result

#  4. SETUP MODELS 
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_opt = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM, num_hands=2, result_callback=h_cb)

pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.LIVE_STREAM, result_callback=p_cb)

face_opt = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/chann/major_project/model_task_files/face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM, result_callback=f_cb)

#  5. MAIN LOOP 
with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
     mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
     mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:
    
    cap = cv.VideoCapture(0)
    # Set resolution to 640x480 for better performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    current_pred = "waiting..."
    conf = 0.0

    print(" System is Ready. U can Show a sign!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Mirror the frame (Matches how you likely look in the mirror)
        # CRITICAL: This flips "Left" to "Right" visually. 
        # MediaPipe handles the labels, but we rely on the sorting logic below.
        frame = cv.flip(frame, 1) 
        
        # 2. Prepare Image
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        
        # 3. Detect (Async)
        hand_lm.detect_async(mp_img, ts)
        pose_lm.detect_async(mp_img, ts)
        face_lm.detect_async(mp_img, ts)
        
        # 4. Construct Feature Vector (Must match Order: Right, Left, Pose, Face)
        feats = []
        
        #  A. HAND SORTING 
        right_hand_lm, left_hand_lm = None, None
        
        if latest_hands and latest_hands.handedness:
            for idx, hand_meta in enumerate(latest_hands.handedness):
                label = hand_meta[0].category_name # "Right" or "Left"
                landmarks = latest_hands.hand_landmarks[idx]
                
                if label == "Right": right_hand_lm = landmarks
                elif label == "Left": left_hand_lm = landmarks

        # Append Right (0-63) then Left (64-126)
        feats.extend(normalize_and_scale(right_hand_lm))
        feats.extend(normalize_and_scale(left_hand_lm))
        
        #  B. POSE 
        if latest_pose and latest_pose.pose_landmarks:
            # Extract Upper Body (11-16)
            subset = [latest_pose.pose_landmarks[0][i] for i in [11,12,13,14,15,16]]
            feats.extend(normalize_and_scale(subset))
        else:
            feats.extend([0.0] * 18)
            
        #  C. FACE 
        if latest_face and latest_face.face_landmarks:
            # Extract Mouth (0, 13, 14, 78, 308)
            subset = [latest_face.face_landmarks[0][i] for i in [0, 13, 14, 78, 308]]
            feats.extend(normalize_and_scale(subset))
        else:
            feats.extend([0.0] * 15)
            
        # 5. Sliding Window
        sequence_buffer.append(feats)
        frame_cnt += 1
        
        # 6. Prediction (Throttled)
        if len(sequence_buffer) == SEQ_LEN and frame_cnt % PREDICTION_FREQ == 0:
            # Prepare Input: (1, 90, 159)
            input_tensor = np.expand_dims(list(sequence_buffer), axis=0)
            
            # Predict
            res = model.predict_on_batch(input_tensor)[0]
            idx = np.argmax(res)
            conf = res[idx]
            
            # Update Label if confident
            if conf > confidence_threshold:
                current_pred = label_map[idx]
            else:
                current_pred = "..." # Reset if unsure
            
        # 7. Visualization
        # Color changes based on confidence (Red=Low, Green=High)
        color = (0, 255, 0) if conf > confidence_threshold else (0, 0, 255)
        
        cv.rectangle(frame, (0,0), (640, 50), (30, 30, 30), -1)
        cv.putText(frame, f"Pred: {current_pred}", (10, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv.putText(frame, f"{int(conf*100)}%", (500, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv.imshow("Sign Language Translator", frame)
        if cv.waitKey(1) == ord('q'): break

    cap.release()
    cv.destroyAllWindows()