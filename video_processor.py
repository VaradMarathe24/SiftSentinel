import cv2
import numpy as np
import tempfile
import os
import torch

def extract_frames(video_path, frame_skip=5, max_frames=30):
    """
    Extract frames from video efficiently.
    """
    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        count += 1

        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


def preprocess_frame(frame):
    """
    Convert frame to model-ready format.
    """
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
    return frame


def predict_video(model, video_path, device):
    """
    Main video prediction pipeline:
    frames → model → average prediction
    """

    frames = extract_frames(video_path)

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    predictions = []

    model.eval()

    with torch.no_grad():
        for frame in frames:
            input_tensor = preprocess_frame(frame)
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

            output = model(input_tensor)

            # sigmoid for binary classification
            prob = torch.sigmoid(output).item()
            predictions.append(prob)

    final_score = sum(predictions) / len(predictions)

    return {
        "frame_predictions": predictions,
        "final_score": final_score,
        "label": "REAL" if final_score > 0.5 else "FAKE"
    }