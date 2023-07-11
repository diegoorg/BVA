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
import easyocr
import numpy as np
#from google.colab.patches import cv2_imshow
from observers import observer_hd, team

#index = 0

# Player ID
def player_id(frame, detections):
    global index
    #cv2_imshow(frame)
    player_id = []
    frame_h, frame_w, _ = frame.shape
    print(frame_h, frame_w)
    for box, _, class_id, tracker_id in detections:
        #print(box)
        # all numbers positive integers
        list1 = np.asarray(box, dtype = 'int')
        x1,y1,x2,y2 = [(i > 0) * i for i in list1]
        #print(x1,x2,y1,y2)
        # hight and width
        h = y2-y1
        w = x2-x1
        # make sure all positive
        a = int(max(0, y1+0.25*h))
        b = int(min(frame_h, y2-0.5*h))
        c = int(max(0, x1+0.25*w))
        d = int(min(frame_w, x2-0.25*w))

        crop = frame[a:b,c:d]

        #img = Image.fromarray(crop, 'RGB')
        #img.save(f"test{index}.jpeg")
        #index += 1

        result = reader.readtext(crop, allowlist = reduced_class)
        print(result)
        if result != []:
            id_box, id_num, id_conf = result[0]
            if id_conf >= ID_THRES:
                player_id.append(id_num)
            else: player_id.append('unk')
        else: player_id.append('unk')

    return player_id

def zipit(detections, player_ids):
    return zip(detections.xyxy, detections.class_id, detections.confidence, detections.tracker_id, player_ids)

# load a YOLOv8 custom model
model = YOLO(f"{HOME}/data/model/y8l-0307.pt") 

# Load OCR model
reader = easyocr.Reader(['en'])
reduced_class = '0123456789'
ID_THRES = 0.9


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

    team_0 = team(0)
    observer = observer_hd(team_0)

    # Detect and track objects using ultralytics SDK
    for result in model.track(source=SOURCE_VIDEO_PATH, tracker="botsort.yaml", save=True, stream=True, agnostic_nms=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Filtering undesired classes
        detections = detections[detections.class_id != 1]
        '''
        # mofify object detection as only one ball handler allowed
        filter_idx = np.where(detections.class_id == 0)[0]
        #print(filter_idx)
        if filter_idx.size > 1:
          filtered_conf = detections.confidence[filter_idx]
          max_idx = filtered_conf.argmax(axis = 0)
          filter_idx = np.delete(filter_idx, max_idx)
          detections.class_id[filter_idx] = 2
          #print(detections.class_id)

        # Feed the player identification
        player_ids = player_id(frame, detections)
        labels_zip = zipit(detections, player_ids)
        #print(list(labels_zip))
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}, ID: {player_id}"
            for _, class_id, confidence, tracker_id, player_id
            in labels_zip
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        '''
        # Feed the player identification
        player_ids = player_id(frame, detections)
        

        # update observers
        observer.upd_observers(detections)
        printers = observer.export_obs()
        labels_zip = zipit(printers, player_ids)
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}, ID: {player_id}"
            for _, class_id, confidence, tracker_id, player_id
            in labels_zip
        ]
        # annotate and display frame
        frame = box_annotator.annotate(scene=frame, detections=printers, labels=labels)
        sink.write_frame(frame)

