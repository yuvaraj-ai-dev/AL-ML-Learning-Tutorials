#!/usr/bin/env python3
"""
Object Detection and Tracking Pipeline Inference Script

This script demonstrates a complete object detection and tracking pipeline
using OpenCV and pre-trained models. It processes video input and outputs
an annotated video with detected and tracked objects.

Usage:
    python inference.py --input_video path/to/video.mp4 --output_video path/to/output.mp4

Requirements:
    - OpenCV
    - NumPy
    - PyTorch (for advanced model)
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

class ObjectDetector:
    """Base class for object detection"""

    def __init__(self, model_type='haar'):
        self.model_type = model_type
        if model_type == 'haar':
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif model_type == 'yolo':
            # For YOLO, you'd load the model here
            # This is a simplified version
            self.net = None
            self.classes = []
        elif model_type == 'faster_rcnn':
            try:
                import torch
                import torchvision
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                from torchvision.transforms import functional as F

                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)

                self.COCO_CLASSES = [
                    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
            except ImportError:
                print("PyTorch not available, falling back to Haar cascades")
                self.model_type = 'haar'
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, frame):
        """Detect objects in frame"""
        if self.model_type == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            detections = []
            for (x, y, w, h) in objects:
                detections.append({
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 1.0,
                    'class': 'face'
                })
            return detections

        elif self.model_type == 'faster_rcnn':
            try:
                import torch
                from torchvision.transforms import functional as F

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    predictions = self.model(img_tensor)

                pred_boxes = predictions[0]['boxes'].cpu().numpy()
                pred_scores = predictions[0]['scores'].cpu().numpy()
                pred_labels = predictions[0]['labels'].cpu().numpy()

                # Filter predictions
                keep = pred_scores > 0.5
                detections = []
                for box, score, label in zip(pred_boxes[keep], pred_scores[keep], pred_labels[keep]):
                    x1, y1, x2, y2 = box.astype(int)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'class': self.COCO_CLASSES[label]
                    })
                return detections
            except:
                # Fallback to Haar
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                objects = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                detections = []
                for (x, y, w, h) in objects:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 1.0,
                        'class': 'face'
                    })
                return detections

        return []

class ObjectTracker:
    """Base class for object tracking"""

    def __init__(self, tracker_type='csrt'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.next_id = 0

    def init_tracker(self, frame, bbox):
        """Initialize a new tracker"""
        if self.tracker_type == 'csrt':
            tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == 'kcf':
            tracker = cv2.TrackerKCF_create()
        else:
            tracker = cv2.TrackerCSRT_create()

        tracker.init(frame, bbox)
        self.trackers.append({
            'tracker': tracker,
            'id': self.next_id,
            'bbox': bbox
        })
        self.next_id += 1

    def update_trackers(self, frame):
        """Update all trackers"""
        active_trackers = []
        for tracker_info in self.trackers:
            success, bbox = tracker_info['tracker'].update(frame)
            if success:
                tracker_info['bbox'] = bbox
                active_trackers.append(tracker_info)

        self.trackers = active_trackers
        return self.trackers

def create_annotated_video(input_video_path, output_video_path, detector, tracker=None, max_trackers=5):
    """Process video and create annotated output"""

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_interval = 30  # Detect every 30 frames

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect objects periodically
        if frame_count % detection_interval == 0:
            detections = detector.detect(frame)

            # Initialize trackers for new detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                bbox = (x1, y1, x2-x1, y2-y1)

                if tracker:
                    tracker.init_tracker(frame, bbox)

        # Update trackers
        tracked_objects = []
        if tracker:
            tracked_objects = tracker.update_trackers(frame)

        # Draw detections/trackings on frame
        if frame_count % detection_interval == 0 and not tracker:
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{detection['class']}:{detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif tracker:
            # Draw tracked objects
            for obj in tracked_objects:
                x, y, w, h = obj['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {obj['id']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='Object Detection and Tracking Pipeline')
    parser.add_argument('--input_video', type=str, default='datasets/sample_videos/sample_clip.mp4',
                        help='Path to input video file')
    parser.add_argument('--output_video', type=str, default='results/annotated_video.mp4',
                        help='Path to output annotated video file')
    parser.add_argument('--detector', type=str, default='haar', choices=['haar', 'faster_rcnn'],
                        help='Detection model to use')
    parser.add_argument('--tracker', type=str, default='csrt', choices=['csrt', 'kcf', 'none'],
                        help='Tracking algorithm to use')

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize detector
    detector = ObjectDetector(model_type=args.detector)

    # Initialize tracker
    tracker = None
    if args.tracker != 'none':
        tracker = ObjectTracker(tracker_type=args.tracker)

    # Process video
    create_annotated_video(args.input_video, args.output_video, detector, tracker)

    print("Inference completed successfully!")

if __name__ == "__main__":
    main()