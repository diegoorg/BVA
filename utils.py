import numpy as np

def channel_img_mean_2(image):
    img_mean = np.mean(image, axis = (0,1))/255
    return img_mean

def decision(th, image):
    img = channel_img_mean_2(image)
    print(img)
    vote_0 = 0
    vote_1 = 0
    for i in range(3):
        if img[i] <= th[i]: vote_0 += 1
        else: vote_1 += 1
    if vote_0 > vote_1:
        return 0
    else: return 1