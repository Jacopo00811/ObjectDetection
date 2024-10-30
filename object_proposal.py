import cv2
from read_XML import read_images_and_xml
import os
import random
import time

start_time = time.time()

# Find the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the current folder path
folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images')
images, annotations = read_images_and_xml(folder_path)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def process_image(image, num_boxes):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    img_copy = image.copy()
    for (x, y, w, h) in rects[:num_boxes]:
        color = random_color()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    return img_copy

# Load three different images
# images_to_process = [images[5], images[10], images[15]]
images_to_process = [images[0]]

# Process each image
for idx, img in enumerate(images_to_process):
    # img_50 = process_image(img, 50)
    # img_100 = process_image(img, 100)
    img_1000 = process_image(img, 1000)

    # Show the images with bounding boxes
    # cv2.imshow(f"Selective Search - Image {idx+1} - 50 Boxes", img_50)
    # cv2.imshow(f"Selective Search - Image {idx+1} - 100 Boxes", img_100)
    cv2.imshow(f"Selective Search - Image {idx+1} - 100 Boxes", img_1000)


end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")
cv2.waitKey(0)
cv2.destroyAllWindows()
