import numpy as np

# Get normalized mean from each color channel for an image
def channel_img_mean_2(image):
    img_mean = np.mean(image, axis = (0,1))/255
    return img_mean

# Classify image based on threshold and feature projection
def decision(image, w_max, th):
    img = channel_img_mean_2(image)
    img_proj = np.dot(w_max, img)
    if img_proj <= th:
        return 1
    else:
        return 0

# Intersection measurement between 2 given bbox
def intersection(bb1, bb2):

    '''
    Code obtained from:
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    '''

    # Check input data
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    return intersection_area

# Integrate player_team and player_id with detections for annotation
def zipit(detections, player_team, player_ids):
    player_id_short = []
    player_team_short = []

    # Append team and ID information for each tracking ID
    for tracker in detections.tracker_id:
        if tracker in player_ids and tracker in player_team:
            player_id_short.append(player_ids[tracker])
            player_team_short.append(player_team[tracker])
        else: 
            player_id_short.append(None)
            player_team_short.append(None)
            
    return zip(detections.xyxy, detections.class_id, detections.confidence, detections.tracker_id, player_team_short, player_id_short)