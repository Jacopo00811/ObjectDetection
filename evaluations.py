import torch
from torchvision.ops import nms, box_iou
from evaluate_proposal import compute_IoU, convert_to_xyxy
import numpy as np
from torchmetrics import AveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Assume vars are called boxes and scores


boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 20, 20]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.75, 0.8], dtype=torch.float32)
iou_threshold = 0.6

detections = torch.tensor([[0, 0, 10, 10, 0.9], [0, 0, 5, 5, 0.75], [0, 0, 20, 20, 0.8]])
ground_truths = torch.tensor([[0, 0, 10, 10, 1], [0, 0, 5, 5, 0], [0, 0, 20, 20, 1]])

# Compare custom compute_IoU and PyTorch's box_iou

gt_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
pred_boxes = torch.tensor([[0.1, 0.1, 1.1, 1.1], [1.1, 1.1, 2.1, 2.1]])

custom_iou = compute_IoU(gt_boxes[0], pred_boxes[0])
print("Original ground truth box (x_min, y_min, width, height):", gt_boxes[0])
print("Original predicted box (x_min, y_min, width, height):", pred_boxes[0])

# Convert to (x_min, y_min, x_max, y_max)
gt_box_single = convert_to_xyxy(gt_boxes[0].unsqueeze(0))  # Add batch dimension
pred_box_single = convert_to_xyxy(pred_boxes[0].unsqueeze(0))  # Add batch dimension

# Print the converted boxes (after unsqueeze and conversion)
print("\nConverted ground truth box (x_min, y_min, x_max, y_max) after unsqueeze:")
print(gt_box_single)
print("Converted predicted box (x_min, y_min, x_max, y_max) after unsqueeze:")
print(pred_box_single)# Use box_iou, which expects 2D tensors (batch_size x 4)
pytorch_iou = box_iou(gt_box_single, pred_box_single)

# PyTorch IoU calculation
# custom_iou = box_iou(ground_truths[:, :4][0], detections[:, :4][0])
# pytorch_iou = box_iou(convert_to_xyxy(ground_truths[:, :4]), convert_to_xyxy(detections[:, :4]))

print("Custom IoU:\n", custom_iou)
print("PyTorch IoU:\n", pytorch_iou)


# Example: binary classification for simplicity
outputs = torch.tensor([[0.8], [0.3], [0.6], [0.1]])  # predicted probabilities (scores)
targets = torch.tensor([1, 0, 1, 0])  # ground truth labels


outputs = outputs.squeeze()
# Initialize the AveragePrecision metric (task='binary' for binary classification)
avg_precision = AveragePrecision(task='binary')

# Update the metric with outputs and targets
avg_precision.update(outputs, targets)

# Compute the average precision score
result = avg_precision.compute()

print(f'Average Precision: {result}')