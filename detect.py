# SETTINGS

import os
HOME = os.getcwd()
print(HOME)

# Check supervision version
import supervision as sv
print("supervision version: ", sv.__version__)

# Check GPU
import torch
print("Working with GPU: ", torch.cuda.is_available())

# Load libraries
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import timeit
import numpy as np
from observers import observer_hd, team
from utils import zipit

# Load an official model
model = YOLO('yolov8l.pt')  
# Load a YOLOv8 custom model
model = YOLO(f"{HOME}/data/model/y8l-2207_02.pt") 

# Video Settings
SOURCE_VIDEO_PATH = f"{HOME}/data/raw_video/test2.mp4"
if os.path.isdir(f"{HOME}/data/raw_video") == False:
  os.mkdir(f"{HOME}/data/raw_video")
TARGET_VIDEO_PATH = f"{HOME}/data/proc_video/test1.mp4"
if os.path.isdir(f"{HOME}/data/proc_video") == False:
  os.mkdir(f"{HOME}/data/proc_video")

# Dict maping class_id to class_name
CLASSES = {'ball':0, 'ball-handler': 1, 'basket':2, 'made-basket':3, 'player':4}
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# Display raw video info
sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)


# INFERENCE

# Create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# Create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=2, text_thickness=2, text_scale=0.8)

# Open target video file
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # Define teams
    teams = {0: team(0),
             1: team(1)}
    # Define observer's handler
    observer = observer_hd(teams)

    # Define variable to measure time
    time_process = 0

    # Detect and track objects using ultralytics SDK
    for result in model.track(source=SOURCE_VIDEO_PATH, tracker="botsort.yaml", save=True, stream=True, agnostic_nms=True):
        
        # Get time for FPS
        time_start = timeit.default_timer()  

        # Get frame for player ID and detections
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        # Add tracking ID
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Update observers
        observer.upd_observers(detections, frame)
        printers = observer.export_obs()
        player_ids = observer.export_ply_id()
        player_team = observer.export_ply_team()
        labels_zip = zipit(printers, player_team, player_ids)

        # Prepare annotations
        labels = []
        for _, class_id, confidence, tracker_id, player_team, player_id in labels_zip:
            if class_id == CLASSES['player'] or class_id == CLASSES['ball-handler']:
                labels.append(f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}, Team {player_team}, ID: {player_id}")
            else:
                labels.append(f"{model.model.names[class_id]} {confidence:0.2f}")
      
        # Annotate frame
        frame = box_annotator.annotate(scene=frame, detections=printers, labels=labels)

        # Report results
        for _, team in teams.items():
            team.report()

        # Write frame
        sink.write_frame(frame)

        # Record time
        time_stop = timeit.default_timer()
        time_current = (time_stop - time_start)*1000
        print('Results processing time: ', time_current)
        time_process += time_current
    
    # Report mean process time
    print("Mean process time (without object detector): ", time_process/video_info.total_frames)

