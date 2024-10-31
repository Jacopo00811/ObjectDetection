from object_proposal import SelectiveSearch, EdgeDetection
import matplotlib.pyplot as plt
from read_XML import read_images_and_xml
import os
import numpy as np

def compute_IoU(box1, box2):
    # box1 and box2 are in the format (x, y, width, height)
    x1_min, x2_min = box1[0], box2[0]
    y1_min, y2_min = box1[1], box2[1]
    x1_max, x2_max = box1[0] + box1[2], box2[0] + box2[2]
    y1_max, y2_max = box1[1] + box1[3], box2[1] + box2[3]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    intersection_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area, box2_area = box1[2] * box1[3], box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area/union_area if union_area > 0 else 0
    return iou


def compute_recall_and_mabo(pred_boxes, gt_boxes, iou_threshold):
    recall = 0
    iou_sum = 0
    matched_boxes = 0

    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = compute_IoU(gt_box, pred_box)
            best_iou = max(best_iou, iou)
        iou_sum += best_iou
        if best_iou >= iou_threshold:
            matched_boxes += 1

    recall = matched_boxes/len(gt_boxes) if gt_boxes else 0
    mabo = iou_sum/len(gt_boxes) if gt_boxes else 0
    return recall, mabo


def evaluate_proposals_on_image(image, gt_boxes, num_boxes, mode, iou_threshold):
    ss_boxes = SelectiveSearch(image, num_boxes, mode)
    edge_boxes = EdgeDetection(image, num_boxes)[0]
    ss_recall, ss_mabo = compute_recall_and_mabo(ss_boxes, gt_boxes, iou_threshold)
    edge_recall, edge_mabo = compute_recall_and_mabo(edge_boxes, gt_boxes, iou_threshold)
    return ss_recall, ss_mabo, edge_recall, edge_mabo


def plot_proposals_for_image(image, gt_boxes, max_num_boxes, mode, iou_threshold):
    boxes_list = list(np.arange(100, max_num_boxes+1, 50))
    ss_recall_list = []
    edge_recall_list = []
    ss_mabo_list = []
    edge_mabo_list = []

    for num_boxes in boxes_list:
        ss_recall, ss_mabo, edge_recall, edge_mabo = evaluate_proposals_on_image(image, gt_boxes, num_boxes, mode, iou_threshold)
        ss_recall_list.append(ss_recall)
        edge_recall_list.append(edge_recall)
        ss_mabo_list.append(ss_mabo)
        edge_mabo_list.append(edge_mabo)
    
    # Plot for Selective Search
    plt.figure(figsize=(10, 5))
    plt.plot(boxes_list, ss_recall_list, color='red', label='SS Recall')
    plt.plot(boxes_list, edge_recall_list, color='blue', label='Edge Detector Recall')
    plt.xlabel('Number of boxes', fontweight='bold')
    plt.ylabel('Recall', fontweight='bold')
    plt.ylim(0, 1.1)
    plt.title('Selective Search Recall vs Number of Boxes', fontsize=20, fontweight="bold", color="red")
    plt.legend()
    plt.savefig('recall_vs_num_boxes.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(boxes_list, ss_mabo_list, color='red', label='SS MABO')
    plt.plot(boxes_list, edge_mabo_list, color='blue', label='Edge Detector MABO')
    plt.xlabel('Number of boxes', fontweight='bold')
    plt.ylabel('MABO', fontweight='bold')
    plt.ylim(0, 1.1)
    plt.title('Selective Search MABO vs Number of Boxes', fontsize=20, fontweight="bold", color="red")
    plt.legend()
    plt.savefig('MABO_vs_num_boxes.png')
    plt.close()


script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'train')
images, annotations = read_images_and_xml(folder_path)
image_index = 0
max_num_boxes = 2500
image = images[image_index]
gt_boxes = annotations[image_index]
plot_proposals_for_image(image, gt_boxes, max_num_boxes, mode='quality', iou_threshold=0.5)

