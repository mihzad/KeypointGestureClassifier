import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import HandLandmarksConnections as mp_hands
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

import numpy as np
import cv2
from scipy.interpolate import interp1d
from pathlib import Path
from PIL import Image
import logging
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import mlflow

FPS=30
VID_LENGTH=16 #frames
MAX_MISSING_FRAMES=4
CONFIDENCE_TRESHOLD=0.7


def draw_landmarks_on_image(rgb_image, detection_result, raw_coords=False):
    margin = 10 # pixels
    if raw_coords:
        hand_landmarks_list = [build_landmark_list(detection_result)]
        #handedness_list = None
    else:
        hand_landmarks_list = detection_result.hand_landmarks
        #handedness_list = detection_result.handedness

    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        #handedness = handedness_list[idx] if handedness_list is not None else None

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = annotated_image.shape
        # x_coordinates = [landmark.x for landmark in hand_landmarks]
        # y_coordinates = [landmark.y for landmark in hand_landmarks]
        # text_x = int(min(x_coordinates) * width)
        # text_y = int(min(y_coordinates) * height) - margin

        # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def build_landmark_list(raw_coords_2d_flattened):
    landmark_list = list()
    
    for x, y in raw_coords_2d_flattened.reshape(-1, 2):
        landmark = NormalizedLandmark()
        landmark.x = x
        landmark.y = y
        landmark.z = 0.0  # safe default
        landmark_list.append(landmark)
    
    return landmark_list


def extract_keypoints(root_dir: Path, output_file: str, logging_file: str, interpolation_kind='cubic', ):
    logging.basicConfig(filename=logging_file, level=logging.INFO, encoding='utf-8')
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.VIDEO,
                                        min_hand_detection_confidence=0.75,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.75,
                                        num_hands=1)

    X = [] # Data
    y = [] # Labels
    failed_vids=[]
    interpolated_vids=[]
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])

    for cls_idx, cls_name in enumerate(classes):
        class_dir = root_dir / cls_name
        
        for vid_dir in class_dir.iterdir():
            if not vid_dir.is_dir(): continue
            
            video_id = f'{cls_name}\\{vid_dir.name}'  # Unique ID for the video
            video_landmarks = [None]*VID_LENGTH
            valid_frames_mask = [False] * VID_LENGTH  # To track which frames are valid
            frame_imgs_bgr = [] #visualization

            detector = vision.HandLandmarker.create_from_options(options)

            
            frame_paths = sorted(list(vid_dir.glob('*.jpg')))[:VID_LENGTH]
            
            for frame_index, img_path in enumerate(frame_paths):
                frame_id = img_path.name
                frame = Image.open(img_path).convert('RGB')
                rgb_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
                
                timestamp_ms = int(frame_index * (1000 / FPS))
                results = detector.detect_for_video(rgb_img, timestamp_ms)

                frame_bgr=cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                frame_imgs_bgr.append(frame_bgr)
                annotated_frame_bgr = draw_landmarks_on_image(frame_bgr, results)
                #cv2.imshow('MediaPipe Hands Recognition', annotated_frame_bgr)


                if results.hand_landmarks:
                    # Check confidence score of the first hand
                    hand_info = results.handedness[0][0]
                    conf = hand_info.score
                    label = hand_info.category_name
                    if conf > CONFIDENCE_TRESHOLD:  # Your adequacy threshold
                        frame_landmarks = np.array([[lm.x, lm.y] for lm in results.hand_landmarks[0]]).flatten() #lm.z is not useful
                        video_landmarks[frame_index] = frame_landmarks
                        valid_frames_mask[frame_index] = True
                
                # if is_success:
                #     success_count += 1
                #     logging.info(f"✅[PASS] Video: \'{video_id}\' | Frame: {frame_id} (Conf: {conf:.2f}, Label: {label})")
                # else:
                #     # Forward fill if possible, else zeros
                #     if len(video_landmarks) > 0 and np.any(video_landmarks[-1]):
                #         frame_landmarks = video_landmarks[-1]
                #         logging.warning(f"⚠️[HALF-FAIL] Video: \'{video_id}\' | Frame: {frame_id} - Low Confidence/No Hand, forward-filled")

                #     else:
                #         logging.warning(f"❌[FAIL] Video: \'{video_id}\' | Frame: {frame_id} - Low Confidence/No Hand")
                #         frame_landmarks = np.zeros(21*2)

                #     if video_id not in failed_vids:
                #         failed_vids.append(video_id)

            invalid_count = sum(not v for v in valid_frames_mask)
            success_count = VID_LENGTH - invalid_count

            if invalid_count == 0:
                X.append(video_landmarks)
                y.append(cls_idx)
                logging.info(f"--- ✅Video {video_id} Complete. Success Rate: {success_count}/{VID_LENGTH} (interpolated {invalid_count} frames) ---")
                continue
            
            if invalid_count > MAX_MISSING_FRAMES:
                failed_vids.append(video_id)
                logging.info(f"---❌ Video {video_id}: skipped due to {invalid_count} missing frames ---\n")
                continue

            # Interpolate missing frames, guaranteed to have at least 16-4=12 valid frames for interpolation
            num_coords = 42  # 21 landmarks * 2 (x,y)
            interpolated_video_landmarks = deepcopy(video_landmarks)
            for coord_idx in range(num_coords):
                known_indices = [i for i, lm in enumerate(video_landmarks) if lm is not None]
                known_values = [video_landmarks[i][coord_idx] for i in known_indices]
                
                interpolate_func = interp1d(x=known_indices, y=known_values, kind=interpolation_kind, 
                                    bounds_error=False, fill_value=(known_values[0], known_values[-1])) #fill_value="extrapolate", too much outliers
                
                ready_vals = interpolate_func(range(VID_LENGTH))

                for i in range(VID_LENGTH):
                    if interpolated_video_landmarks[i] is None:
                        interpolated_video_landmarks[i] = np.zeros(num_coords)
                    interpolated_video_landmarks[i][coord_idx] = ready_vals[i]
            

            #log the interpolation to mlflow
            interpolated_vids.append(video_id)
            for i in range(VID_LENGTH):
                annotated_frame_bgr = draw_landmarks_on_image(frame_imgs_bgr[i], interpolated_video_landmarks[i], raw_coords=True)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                if valid_frames_mask[i]:
                    mlflow.log_image(annotated_frame_rgb, f"interpolated_vids/{vid_dir}/frame_{i}.jpg")
                else:
                    mlflow.log_image(annotated_frame_rgb, f"interpolated_vids//{vid_dir}/frame_{i}_interpolated.jpg")


            # Now use interpolated_landmarks for X.append(...)
            logging.info(f"--- ✅Video {video_id} Complete. Success Rate: {success_count}/{VID_LENGTH} (interpolated {invalid_count} frames) ---")         
            X.append(interpolated_video_landmarks)
            y.append(cls_idx)


    logging.info(f"All videos done. failed videos: {len(failed_vids)}/{len(X)+len(failed_vids)};\n")
    logging.info(f"All videos done. interpolated videos: {len(interpolated_vids)}/{len(X)+len(failed_vids)};\n")

    mlflow.log_param('failed_videos', f"{len(failed_vids)}/{len(X)+len(failed_vids)}")
    mlflow.log_param('interpolated_videos', f"{len(interpolated_vids)}/{len(X)+len(failed_vids)}")
    
    logging.info(f"Failed video names:")
    for i, vid_id in enumerate(failed_vids):
        logging.info(f'   {i}) \"{vid_id}\"')
    
    # Save as compressed numpy files
    np.savez_compressed(output_file, X=np.array(X, dtype=np.float32), y=np.array(y))
    mlflow.log_artifact(output_file, artifact_path=output_file)
    mlflow.log_artifact(logging_file, artifact_path=logging_file)
    print(f"Dataset saved to {output_file} with shape {np.array(X).shape}")

if __name__ == "__main__":
    # params = {
    #     'experiment_name': 'GC Keypoint Extraction',
    #     'run_name': 'keypoint_extraction_train',
    #     'root_dir': Path('data_images/train'),
    #     'output_file': 'data/data_train.npz',
    #     'logging_file': 'extraction_log_train.txt',
    #     "interpolation_kind": 'cubic'
    # }
    params = {
        'experiment_name': 'GC Keypoint Extraction',
        'run_name': 'keypoint_extraction_validation',
        'root_dir': Path('data_images/val'),
        'output_file': 'data/data_validation.npz',
        'logging_file': 'extraction_log_validation.txt',
        "interpolation_kind": 'cubic'
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params['experiment_name'])


    with mlflow.start_run(run_name=params['run_name']):
        mlflow.log_params(params)
        extract_keypoints(params['root_dir'], params['output_file'], params['interpolation_kind'])