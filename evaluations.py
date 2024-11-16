import torch
from torchvision.ops import nms, box_iou
from evaluate_proposal import compute_IoU, convert_to_xyxy
import numpy as np
from torchmetrics import AveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Assume vars are called boxes and scores


boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 20, 20]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.75, 0.8], dtype=torch.float32)
iou_threshold = 0.9

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


# # Example: binary classification for simplicity
# outputs = torch.tensor([[0.8], [0.3], [0.6], [0.1]])  # predicted probabilities (scores)
# targets = torch.tensor([1, 0, 1, 0])  # ground truth labels


# outputs = outputs.squeeze()
# # Initialize the AveragePrecision metric (task='binary' for binary classification)
# avg_precision = AveragePrecision(task='binary')

# # Update the metric with outputs and targets
# avg_precision.update(outputs, targets)

# # Compute the average precision score
# result = avg_precision.compute()

# print(f'Average Precision: {result}')



boxes = torch.tensor([[1, 2, 5, 6], [2, 3, 6, 7], [10, 10, 15, 15]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.2, 0.8], dtype=torch.float32)

# Compute the IoU of boxes[0] and boxes[1]
iou_boxes_0_1 = box_iou(boxes[0].unsqueeze(0), boxes[1].unsqueeze(0))
print("IoU of boxes[0] and boxes[1]:", iou_boxes_0_1)
# scores = torch.rand((32, 1))
scores = scores.squeeze()
# print("Coordinates tensor:\n", coords)

iou_threshold = 0.4
# # Apply non-maximum suppression
keep = nms(boxes, scores, iou_threshold)
print("Indices to keep after NMS:\n", keep)

# def non_max_suppression(boxes, scores, iou_threshold):
#     # Convert to numpy for easier manipulation
#     # boxes_np = boxes.numpy()
#     # scores_np = scores.numpy()
#     boxes_np = boxes
#     scores_np = scores
#     print(boxes_np)
#     print(scores_np)

#     sorted_tensor, indices = torch.sort(scores_np, descending=True)

#     # Get the indices of boxes sorted by scores (highest first)
#     # indices = np.argsort(scores_np)[::-1]

#     keep = []
#     while len(indices) > 0:
#         current = indices[0]
#         keep.append(current)
#         if len(indices) == 1:
#             break

#         current_box = boxes_np[current]
#         remaining_boxes = boxes_np[indices[1:]]

#         # Compute IoU of the current box with the remaining boxes
#         ious = compute_IoU(current_box, remaining_boxes)

#         # Select boxes with IoU less than the threshold
#         indices = indices[1:][ious < iou_threshold]

#     return torch.tensor(keep)

# # Use the custom NMS function
# keep_custom = non_max_suppression(boxes, scores, iou_threshold)
# print("Indices to keep after custom NMS:\n", keep_custom)


# ground_t = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1]])
ground_t = torch.tensor([1, 0, 1, 1, 1, 1, 0, 1])
# pred_t = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1]])

# ground_t = ground_t.squeeze()
pred_t = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1])

# Perform softmax on the predictions
ground_t = torch.softmax(ground_t.float(), dim=0)
# print(f"Ground truth after softmax: {ground_t}")
# pred_t = torch.softmax(pred_t.float(), dim=0)


# # Average precision
avg_precision = AveragePrecision(task='binary')

# # Update the metric with outputs and targets
avg_precision.update(ground_t, pred_t)

# # Compute the average precision score
result = avg_precision.compute()

print(f'New Average Precision: {result}')






# Example: binary classification for simplicity
# outputs = torch.tensor([[0.8], [0.3], [0.6], [0.1], [0.5], [0.7], [0.2], [0.9]])  # predicted probabilities (scores)
targets = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1])  # ground truth labels
outputs = torch.tensor([0.8, 0.3, 0.6, 0.1, 0.5, 0.7, 0.2, 0.9])

outputs = torch.tensor([0.4969, 0.4787, 0.4823, 0.5447, 0.4930, 0.5145, 0.4438, 0.5660, 0.5115,
                        0.5227, 0.5076, 0.4992, 0.5354, 0.5437, 0.5199, 0.4411, 0.5829, 0.5108,
                        0.5514, 0.5957, 0.5347, 0.4814, 0.4674, 0.5365, 0.4649, 0.5281, 0.5798,
                        0.3872, 0.5550, 0.4741, 0.4909, 0.4619])

outputs = torch.nn.functional.softmax(outputs, dim=0)
print(outputs)
# Generate targets of the same length as outputs with random 1s and 0s
targets = torch.randint(0, 2, outputs.shape)
print(targets)
# outputs = outputs.squeeze()
# Initialize the AveragePrecision metric (task='binary' for binary classification)
avg_precision = AveragePrecision(task='binary')

# Update the metric with outputs and targets
avg_precision.update(outputs, targets)

# Compute the average precision score
result = avg_precision.compute()

print(f'Average Precision: {result}')

