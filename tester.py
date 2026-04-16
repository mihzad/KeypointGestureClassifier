import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import HandLandmarksConnections as mp_hands
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from PIL import Image

from helpers.common import curr_time
FPS = 30
VID_LENGTH = 16
MP_CONFIDENCE_TRESHOLD = 0.5
GC_CONFIDENCE_TRESHOLD = 0.5
NO_HAND_DETECTED_LABEL = 34


def collate_fn(batch):
    vid_frames, labels = zip(*batch)

    return vid_frames, torch.tensor(labels)

def perform_testing(GC, batch_size, testing_set):
    print("performing testing...")
    testing_loader = DataLoader(testing_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=6)

    mp_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    mp_options = vision.HandLandmarkerOptions(base_options=mp_base_options,
                                        running_mode=vision.RunningMode.VIDEO,
                                        min_hand_detection_confidence=0.75,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.75,
                                        num_hands=1)

    GC.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        mp_not_detected_count = 0
        gc_not_detected_count = 0
        targets = []
        predictions = []

        video_index = 0
        vid_count = len(testing_set)
        for inputs, labels in testing_loader: #inputs is list of vids = lists of PIL images.
            #inputs, labels = inputs.cuda(), labels.cuda()

            outputs = [None] * len(inputs) #will be filled with landmark data extracted from the videos. If no hand detected in a video - corresponding output will be None, and that video will be skipped during training.
            for batch_index in range(len(inputs)):
                
                print(f"[{curr_time()}] processing video {video_index+1}/{vid_count}...")
                video_index += 1

                video_landmarks = []
                mp_detector = vision.HandLandmarker.create_from_options(mp_options)

                for frame_index, frame_img in enumerate(inputs[batch_index]): #inputs = tuple of (list of vid_frames)
                    #frame_img is PIL image.
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame_img))
                    timestamp_ms = int(frame_index * (1000 / FPS))
                    frame_results = mp_detector.detect_for_video(mp_img, timestamp_ms)
                    if frame_results.hand_landmarks:
                        # Check confidence score of the first hand
                        hand_info = frame_results.handedness[0][0]
                        conf = hand_info.score
                        if conf > MP_CONFIDENCE_TRESHOLD:
                            frame_landmarks = np.array([[lm.x, lm.y] for lm in frame_results.hand_landmarks[0]]).flatten()
                            video_landmarks.append(frame_landmarks)
                            continue
                    # If no hand detected or confidence is low - outputs for the vid will be None & video skipped.
                    outputs[batch_index] = NO_HAND_DETECTED_LABEL
                    mp_not_detected_count += 1
                    break
                if outputs[batch_index] == NO_HAND_DETECTED_LABEL:
                    total += 1
                    continue


                keypoint_inputs = torch.tensor(np.array(video_landmarks), dtype=torch.float32, device=torch.device("cuda"))#(T, 42) - 21 landmarks with x,y for each
                gc_logits = GC(keypoint_inputs.unsqueeze(0)) #(1,T,42) -> (1,N_CLASSES)
                gc_probs = F.softmax(gc_logits, dim=1)

                pred_vals, pred_classes = torch.max(gc_probs.data, 1)
                if pred_vals > GC_CONFIDENCE_TRESHOLD:
                    outputs[batch_index] = pred_classes.item()
                    correct += (outputs[batch_index] == labels[batch_index].item())
                else:
                    outputs[batch_index] = NO_HAND_DETECTED_LABEL
                    gc_not_detected_count += 1
                total += 1

            

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(outputs)

        print(f"Final accuracy: {correct/total:.4f} ({correct}/{total})")
        print(f"MP not detected: {mp_not_detected_count}/{total-correct}")
        print(f"GC not detected: {gc_not_detected_count}/{total-correct}")
        print(f"GC misclassified: {total-correct-mp_not_detected_count-gc_not_detected_count}/{total-correct}")

        targets, predictions = np.array(targets), np.array(predictions)
        print(f"Targets: min = {np.min(targets)}, max = {np.max(targets)}")
        print(f"Predictions: min = {np.min(predictions)}, max = {np.max(predictions)}")

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)
    class_names = [
        'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є',
        'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л',
        'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
        'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', '-'
    ]
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots()
    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax, text_kw={'size': 6})
    plt.show()

    print(classification_report(y_true=targets, y_pred=predictions, target_names=class_names))
