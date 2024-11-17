import os
from torchmetrics import AveragePrecision
from tqdm import tqdm
from dataset import CroppedProposalDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torchsummary import summary
from torch.utils.tensorboard.writer import SummaryWriter
from evaluate_proposal import convert_to_xyxy
from model import MultiModel
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
from torchvision.ops import nms
import argparse
import cv2
import random
import itertools
import json

from read_XML import read_images_and_xml, read_xml_gt_og
 
 # Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()




def save_hyperparameters_to_txt(hyperparameters, file_path):
    """
    Save hyperparameters dictionary to a .txt file in JSON format.

    Args:
        hyperparameters (dict): Dictionary of hyperparameters.
        file_path (str): Path to the file where hyperparameters will be saved.

    Example:
        save_hyperparameters_to_txt(hyperparameters, "hyperparameters.txt")
    """
    try:
        with open(file_path, 'w') as f:
            # Convert dictionary to a formatted JSON string and write to the file
            json.dump(hyperparameters, f, indent=4)
        print(f"Hyperparameters saved successfully to {file_path}")
    except Exception as e:
        print(f"Failed to save hyperparameters: {e}")


# Function to randomly sample hyperparameters
def sample_hyperparameters(hyperparameter_grid, num_samples):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for key, values in hyperparameter_grid.items():
            sample[key] = random.choice(values)
        samples.append(sample)
    return samples

# Function to create all combinations of hyperparameters
def create_combinations(hyperparameter_grid):
    keys, values = zip(*hyperparameter_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)
 
def train_net(model, logger, hyper_parameters, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, directory):
    optimizer, scheduler = set_optimizer_and_scheduler(hyper_parameters, model)
 
    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0
 
    images,labels = [], []
 
    for epoch in range(epochs):  # loop over the dataset multiple times
 
        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')
 
        training_loss = 0
        model.train()  # Set the model to training mode
        train_losses = []
        accuracies = []
       
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the parameter gradients for the current minibatch iteration
 
 
            # coords(xmin, ymin, xmax, ymax, label, index)
            images, labels, xml_dir, _ = batch
            images = images.squeeze(0)  # Remove the batch dimension when batch_size is 1
            labels = labels.squeeze(0)  # Remove the batch dimension when batch_size is 1
            xml_dir = xml_dir[0]  # Remove the tuple
 
            labels = labels.to(device)
            images = images.to(device)
 
            # Forward pass, backward pass and optimizer step
            predicted_labels = model(images) # .squeeze(1)# [:,1]


            # Forward pass, backward pass and optimizer step
            labels = labels.unsqueeze(dim=1) # [32,1]
            predicted_labels = model(images)#.squeeze(1)# [:,1]
            
            if labels.shape != (32, 1) or predicted_labels.shape != (32, 1):
                print(f"Breaking loop because one of the tensors has an incorrect shape. ")
                    # f"labels shape: {labels.shape}, predicted_labels shape: {predicted_labels.shape}")
                break  # Exit the loop if shapes are not as expected
            else:
                labels = labels.float()
                # print(f"Logits: {torch.logit(predicted_labels)} and shape: {torch.logit(predicted_labels).shape}")
                # print(f"Logits: {(predicted_labels)}")

                # predicted_labels = F.sigmoid(predicted_labels)
            
            # small_values_tensor = torch.randn(32, 1, requires_grad=True) * 1e-7  # Even smaller values
            # small_values_tensor = small_values_tensor.to(device)
            # print(f"Small values: {small_values_tensor}")



            # loss_train = loss_function(labels, (predicted_labels))
            loss_train = loss_function(predicted_labels, labels)
            

            # loss_train = loss_function(labels, small_values_tensor)

            # print(f"Loss train 1: {loss_train}")
            loss_train.backward()
            # print(f"Loss train 2: {loss_train}")
            optimizer.step()
            # Accumulate the loss and calculate the accuracy of predictions
            # print(f"Loss train item: {loss_train.item()}")

            training_loss += loss_train.item()
            # print(f"training_loss: {training_loss}")
            train_losses.append(loss_train.item())
 
            # Running train accuracy
            predicted_labels = F.sigmoid(predicted_labels)
            predicted_labels = (predicted_labels > 0.5).int()
 
            # _, predicted = predicted_labels.max(1)
            num_correct = (predicted_labels == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)
 
            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))
 
            logger.add_scalar(f'Train loss', loss_train.item(
            ), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'Train accuracy', train_accuracy, epoch*len(dataloader_train)+train_iteration)
        all_train_losses.append(sum(train_losses)/len(train_losses))
        all_accuracies.append(sum(accuracies)/len(accuracies))
 
        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        val_losses = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels, xml_dir, coords = batch
                images = images.squeeze(0)  # Remove the batch dimension when batch_size is 1
                labels = labels.squeeze(0)  # Remove the batch dimension when batch_size is 1
                xml_dir = xml_dir[0]  # Remove the tuple
                   
                images = images.to(device)
                labels = labels.to(device)
 
                # Forward pass
                output = model(images)#.squeeze(1)
                labels = labels.unsqueeze(dim=1)

                if labels.shape != (32, 1) or output.shape != (32, 1):
                    print(f"Breaking loop because one of the tensors has an incorrect shape. "
                    f"labels shape: {labels.shape}, output shape: {output.shape}")
                    break  # Exit the loop if shapes are not as expected
                else:
                    labels = labels.float()
                    # output = F.sigmoid(output)
                # TODO:
                # Calculate the loss
                # loss_val = loss_function(labels, output)
                loss_val = loss_function(output, labels)
 
                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())
 
                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))
               
                # Update the tensorboard logger.
                logger.add_scalar(f'Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
                # If you want to log the validation accuracy, you can do it here.
            all_val_losses.append(sum(val_losses)/len(val_losses))
 
        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)
 
        logger.add_scalars(f'Combined', {'Validation loss': validation_loss,
                                                 'Train loss': training_loss/len(dataloader_train)}, epoch)
        if scheduler is not None:
            scheduler.step()
            print(f"Current learning rate: {scheduler.get_last_lr()}")
 
    if scheduler is not None:
        logger.add_hparams(
            {f"Step_size": scheduler.step_size, f'Batch_size': 32, f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    else:
        logger.add_hparams(
            {f"Step_size": "None", f'Batch_size': 32, f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
   
   
   

    # Check accuracy and save model
    accuracy, ap = check_accuracy(model, dataloader_test, device, hyperparameters=hyper_parameters)
    save_dir = os.path.join(directory, f'ap_{ap:.3f}.pth')
    torch.save(model.state_dict(), save_dir)
 
    return accuracy, ap
 
def set_optimizer_and_scheduler(new_hp, model):
    if new_hp["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=new_hp["learning rate"],
                                     betas=(new_hp["beta1"],
                                            new_hp["beta2"]),
                                     weight_decay=new_hp["weight decay"],
                                     eps=new_hp["epsilon"])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=new_hp["learning rate"],
                                    momentum=new_hp["momentum"],
                                    weight_decay=new_hp["weight decay"])
    if new_hp["scheduler"] == "Yes":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    else:
        scheduler = None
    return optimizer, scheduler
 
def check_accuracy(model, dataloader, device, save_dir=None, hyperparameters=None):
    model.eval()
    num_correct = 0
    num_samples = 0
    y_true = []
    y_pred = []
    misclassified = []
    all_predictions = []
    all_ground_truths = []
    plots = {}

    with torch.no_grad():
        for data in dataloader:
            # coords(xmin, ymin, xmax, ymax, SS_label, index)
            images, labels, xml_dir, coords = data
            images = images.squeeze(0)  # Remove the batch dimension when batch_size is 1
            labels = labels.squeeze(0)  # Remove the batch dimension when batch_size is 1
            # convert labels to int
            labels = labels.int()
            xml_dir = xml_dir[0]  # Remove the tuple

            ground_truth_boxes = read_xml_gt_og(xml_dir)



            # all_ground_truths.extend(ground_truth_boxes)

 
            images = images.to(device)
            labels = labels.to(device)
            coords = coords.to(device)
 
            # TODO: IMPLEMENT HERE THE TEST TIME PROCEDURE
 
            scores = model(images)#.squeeze()  
            labels = labels.unsqueeze(dim=1)

          
            labels = labels.float()
            scores = F.sigmoid(scores)  
            

            if scores is not None and scores.numel() > 0:
                mask = (scores > 0.5).int()


            mask = mask.squeeze(1)
            coords = coords.squeeze(0)


            selected_coords = coords[mask == 1]
            # scores = F.softmax(scores, dim=1)
            pos_scores = scores[mask == 1]

            # Apply NMS:
           
            boxes = selected_coords[:, :4].to(device).float()

            # # Ensure boxes and pos_scores are 2D tensors
            # if boxes.dim() == 3:
            #     boxes = boxes.squeeze()
            # elif boxes.dim() == 1:
            #     boxes = boxes.unsqueeze()

            # if pos_scores.dim() == 2:
            #     pos_scores = pos_scores.squeeze()
            #     keep = nms(boxes, pos_scores, iou_threshold=hyperparameters['iou_threshold'])
            # elif pos_scores.dim() == 1:
            #     keep = nms(boxes, pos_scores, iou_threshold=hyperparameters['iou_threshold'])
            # elif pos_scores.dim() == 0:
            #     continue # Skip if no scores are selected as positive
            # Apply NMS
            if boxes.numel() > 0 and pos_scores.numel() > 0:
                boxes = boxes.view(-1, 4)  # Ensure boxes shape is [N, 4]
                pos_scores = pos_scores.view(-1)  # Ensure pos_scores shape is [N]
                
                keep = nms(boxes, pos_scores, iou_threshold=hyperparameters['iou_threshold'])
                
                if len(keep) == 0:
                    print("No boxes selected by NMS, skipping.")
                    continue

                all_predictions.extend(pos_scores[keep].cpu().numpy())
                all_ground_truths.extend(labels[keep].cpu().numpy())
            else:
                # print("No valid boxes or scores, skipping this batch.")
                continue

            

            if pos_scores[keep].dim() == 0:
                # skip if no boxes are selected and continue to the next image
                continue
            else:
                all_predictions.extend(pos_scores[keep].cpu().numpy())
                all_ground_truths.extend(labels[keep].cpu().numpy())

            if len(plots) < 6:
                plots[xml_dir] = (xml_dir, labels[keep], pos_scores[keep], keep, boxes[keep])

            # Calculating validation metric stuff ...

            predictions = (scores.squeeze() > 0.5).int()
            # predictions = labels


            num_correct += (predictions == labels.squeeze()).sum().item()
            num_samples += len(predictions)
           
            # Save predictions and labels
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
 

    avg_precision = AveragePrecision(task='binary')


    # cast all_ground_truths to int tensor

    all_predictions = torch.tensor(np.array(all_predictions).flatten()).float()
    all_ground_truths = torch.tensor(np.array(all_ground_truths).flatten()).int()
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        # avg_precision.update(torch.tensor(all_predictions), all_ground_truths)
        avg_precision.update(all_predictions.clone().detach(), all_ground_truths.clone().detach())

        results = avg_precision.compute()
    else:
        print("No valid predictions or ground truths for this batch.")
        results = 0.0


    # avg_precision.update(torch.tensor(all_predictions), all_ground_truths)
    # results = avg_precision.compute()
    print(f"Average Precision: {results}")
    accuracy = float(num_correct)/float(num_samples)
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy * 100:.3f}%")
    
    classes = ('Background', 'Positive') # TODO: CHECK IF THIS IS CORRECT


 
    model.train()
    return accuracy, results
 


def rescale_0_1(image):
    """Rescale pixel values to range [0, 1] for visualization purposes only."""
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image-min_val)/abs(max_val-min_val)
    return rescaled_image
 

def hyperparameter_search(hyperparameters, hparam_grid, device, loss_function, dataloader_train, dataloader_validation, dataloader_test):

    

    run_dir = "Results"
    os.makedirs(run_dir, exist_ok=True)
    modeltype = hyperparameters['backbone']


    best_performance = 0
    best_hyperparameters = None
    run_counter = 0

    for hparams in hparam_grid:
        

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Current Hyperparameters: {hparams}")
        hparams = {**hyperparameters, **hparams}


        model = MultiModel(backbone=hparams['backbone'], hyperparameters=hparams, load_pretrained=True).to(device)

        epochs = hparams['epochs']
        modeltype_directory = os.path.join(run_dir, f'{modeltype}_{epochs}')
        os.makedirs(modeltype_directory, exist_ok=True)

        log_dir = os.path.join(modeltype_directory, f'{hparams["network name"]}_{hparams["optimizer"]}_Scheduler_{hparams["scheduler"]}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)


        accuracy, ap = train_net(model, logger, hparams, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, modeltype_directory)

        run_counter += 1

        # Update best hyperparameters 
        if ap > best_performance:
            best_performance = ap
            best_hyperparameters = hparams

        logger.close()

    print("######### Finished Hyperparameter Search! ########")

    return best_hyperparameters



 
################### MAIN CODE ###################

 
hyperparameters = {
    'learning rate' : 1e-3,
    'step size' : 30,
    'momentum': 0.9,
    'optimizer': 'Adam',
    'number of classes': 1,
    'device': 'cuda',
    'image size': 256,
    'backbone': 'resnet152', # "mobilenet_v3_large" or "resnet152"
    'torch home': 'TorchvisionModels',
    'network name': 'Test-0',
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-08,
    'number of workers': 3,
    'weight decay': 0.0005,
    'scheduler': 'Yes',
    'iou_threshold': 0.5,
    'epochs': 200
}

hyperparameter_grid = {
    'step size' : [10, 20, 30],
    'learning rate' : [1e-2, 5e-3, 1e-3, 1e-4],
    'gamma' : [0.7, 0.8, 0.9, 0.95],
    'weight decay' : [1e-5, 1e-4, 1e-3, 1e-2]

}
 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet152', help='The backbone of the model')
    parser.add_argument('--epochs', type=int, default='100', help='Number of epochs')

    args = parser.parse_args()
    hyperparameters['backbone'] = args.backbone
    hyperparameters['epochs'] = args.epochs

    print(f"Conducting HPSearch for {hyperparameters['backbone']} model")
     
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    if torch.cuda.is_available():
        print("This code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, saturation=0.3),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomAdjustSharpness(2, p=0.5),
        # Do not include the risezing and ToTensor transforms here
    ])


    loss_function = torch.nn.BCEWithLogitsLoss()

    train_dataset = CroppedProposalDataset('train', transform=transform, size=hyperparameters['image size'])
    print(f"Created a new Dataset for training of length: {len(train_dataset)}")
    val_dataset = CroppedProposalDataset('val', transform=transform, size=hyperparameters['image size'])
    print(f"Created a new Dataset for validation of length: {len(val_dataset)}")
    test_dataset = CroppedProposalDataset('test', transform=None, size=hyperparameters['image size'])
    print(f"Created a new Dataset for testing of length: {len(test_dataset)}")
    
    models_folder_path = os.path.join(script_dir, 'TorchvisionModels')
    os.environ['TORCH_HOME'] = models_folder_path
    os.makedirs(models_folder_path, exist_ok=True)
   
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for training")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for validation")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for testing")
    

    
    # accuracy, ap = train_net(model, logger, hyperparameters, device,
    #                             loss_function, train_loader, val_loader, test_loader, modeltype_directory)
    # print(f"Final accuracy: {accuracy}")
    
    samples = create_combinations(hyperparameter_grid)
    print(f"Combinations to test: {len(samples)}")
    best_hyperparameters = hyperparameter_search(hyperparameters, samples, device,
                          loss_function, train_loader, val_loader, test_loader)
    
    # Save the best hyperparameters to a txt file

    save_hyperparameters_to_txt(best_hyperparameters, f"HPResults/Best-{hyperparameters['backbone']}-params.txt")


