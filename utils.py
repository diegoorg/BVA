import numpy as np

def channel_img_mean_2(image):
    img_mean = np.mean(image, axis = (0,1))/255
    return img_mean

def decision(th, image):
    img = channel_img_mean_2(image)
    #print(img)
    vote_0 = 0
    vote_1 = 0
    for i in range(3):
        if img[i] <= th[i]: vote_0 += 1
        else: vote_1 += 1
    if vote_0 > vote_1:
        return 0
    else: return 1

def intersection(bb1, bb2):

        # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]

        # determine the coordinates of the intersection rectangle
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