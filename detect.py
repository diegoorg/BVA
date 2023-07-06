import os
HOME = os.getcwd()
print(HOME)

import supervision as sv
print("supervision.__version__:", sv.__version__)

# Check GPU
import torch
print('Working with GPU:')
print(torch.cuda.is_available())

# Load libraries
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from tqdm.notebook import tqdm
import numpy as np

# load a custom model
model = YOLO(f"{HOME}/data/model/y8l-0307.pt") 


# SETTINGS
SOURCE_VIDEO_PATH = f"{HOME}/data/raw_video/test1.mp4"
if os.path.isdir(f"{HOME}/data/raw_video") == False:
  os.mkdir(f"{HOME}/data/raw_video")
TARGET_VIDEO_PATH = f"{HOME}/data/proc_video/test1.mp4"
if os.path.isdir(f"{HOME}/data/proc_video") == False:
  os.mkdir(f"{HOME}/data/proc_video")

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# class_ids of interest - player and ball-handler
CLASS_ID = [0,2]

sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)


# INFERENCE
# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=2, text_thickness=2, text_scale=1)

# open target video file
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # Detect and track objects using ultralytics SDK
    for result in model.track(source=SOURCE_VIDEO_PATH, tracker="botsort.yaml", save=True, stream=True, agnostic_nms=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Filtering undesired classes
        detections = detections[detections.class_id != 1]

        # mofify object detection as only one ball handler allowed
        filter_idx = np.where(detections.class_id == 0)[0]
        print(filter_idx)
        if filter_idx.size > 1:
          filtered_conf = detections.confidence[filter_idx]
          max_idx = filtered_conf.argmax(axis = 0)
          filter_idx = np.delete(filter_idx, max_idx)
          detections.class_id[filter_idx] = 2
          print(detections.class_id)

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # annotate and display frame
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        sink.write_frame(frame)

