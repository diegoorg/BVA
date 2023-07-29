import easyocr
import os

# Load OCR model
reader = easyocr.Reader(['en'])

# Image folder 
HOME = os.getcwd()
folder = f"{HOME}/data/test_easyOCR/"
reduced_class = '0123456789'

# Initialize variables for analysis
im_total = 0
im_correct = 0
im_conf = 0
im_detec = 0
im_conf_inc = 0

# Loop files
for path in os.listdir(folder):
    image_name = path.split('.')
    if image_name[-1] != 'jpg':
        continue

    im_total += 1

    image_name = image_name[0]
    image_label = image_name.split('_')
    image_label = image_label[-1]
    print(image_name, image_label)

    # Model inference
    full_path = folder + path
    result = reader.readtext(full_path, allowlist = reduced_class)
    print(result)
    if result != []:
        im_detec += 1
        if result[-1][-2] == image_label:
            im_correct += 1
            im_conf += result[-1][-1]
        else:
            im_conf_inc += result[-1][-1]

# Calculate variables
im_correct_perc = im_correct/im_total * 100
im_detect_perc = im_detec/im_total * 100
im_correct_detect_perc = im_correct/im_detec * 100
im_correct_conf_mean = im_conf/im_correct
im_incorrect_conf_mean = im_conf_inc/(im_detec - im_correct)

# Print variables
print('Correct detections', im_detect_perc)
print('Correct recognitions among detections', im_correct_detect_perc)
print('Correct recognitions total', im_correct_perc)
print('Mean confidence of correct recognitions', im_correct_conf_mean)
print('Mean confidence of incorrect recognitions', im_incorrect_conf_mean)