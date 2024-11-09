import cv2
from read_XML import draw_annotations, read_images_and_xml
import os
import random
import numpy as np

# script_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
# images, annotations, names = read_images_and_xml(folder_path)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def SelectiveSearch(image, num_boxes, mode='quality'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects[:num_boxes]


def ShowSelectiveSearch(image, num_boxes, mode='quality'):
    rects = SelectiveSearch(image, num_boxes, mode)
    img_copy = image.copy()
    for (x, y, w, h) in rects:
        color = random_color()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    return img_copy


def EdgeDetection(image, num_boxes, model_path='model/model.yml'):
    model_full_path = os.path.join(os.path.dirname(__file__), model_path)
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model_full_path)
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes(alpha=0.9, beta=0.7)
    edge_boxes.setMaxBoxes(num_boxes)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    return boxes


def ShowEdgeDetection(image, num_boxes):
    boxes = EdgeDetection(image, num_boxes)[0] # Take the first element of the tuple which is the boxes
    img_copy = image.copy()
    for b in boxes:
        x, y, w, h = b
        color = random_color()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    return img_copy


# image_index = 0
# num_boxes = 20
# output_path_ss = os.path.join(os.path.dirname(__file__), f"SS_image_{image_index}.jpg")
# cv2.imwrite(output_path_ss, ShowSelectiveSearch(images[image_index], num_boxes, mode='quality'))
# output_path_ed = os.path.join(os.path.dirname(__file__), f"ED_image_{image_index}.jpg")
# cv2.imwrite(output_path_ed, ShowEdgeDetection(images[image_index], num_boxes))
# draw_annotations(images, annotations, image_index)