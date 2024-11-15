import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_boxes(image, boxes, labels=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i, box in enumerate(boxes):
        # box format: (xmin, ymin, xmax, ymax)
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if labels:
            plt.text(box[0], box[1], labels[i], color='white', fontsize=12, backgroundcolor='red')
    
    plt.show()

def compute_iou(box1, box2):
    # box1 and box2 format: (xmin, ymin, xmax, ymax)
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

# Example usage
if __name__ == "__main__":
    image = np.ones((30, 30, 3))  # Create a blank white image
    # boxes = [[1, 2, 5, 6], [2, 3, 6, 7]]  # (xmin, ymin, xmax, ymax)
    boxes = [[1, 2, 5, 6], [2, 3, 5, 8]]  # (xmin, ymin, xmax, ymax)
    labels = ['Box 1', 'Box 2']
    
    plot_boxes(image, boxes, labels)
    iou = compute_iou(boxes[0], boxes[1])
    print(f'IoU: {iou}')