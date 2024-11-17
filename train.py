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

from read_XML import read_images_and_xml, read_xml_gt_og
 
 # Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()


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
                    output = F.sigmoid(output)
                # TODO:
                # Calculate the loss
                loss_val = loss_function(labels, output)
 
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
    accuracy, ap = check_accuracy(model, dataloader_test, device)
    save_dir = os.path.join(directory, f'ap_{ap:.3f}.pth')
    torch.save(model.state_dict(), save_dir)
 
    return accuracy
 
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
 
def check_accuracy(model, dataloader, device, save_dir=None):
    model.eval()
    num_correct = 0
    num_samples = 0
    y_true = []
    y_pred = []
    misclassified = []
    all_predictions = []
    all_ground_truths = []
    

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
           
            boxes = selected_coords[:, :4].to(device).squeeze().float()



            keep = nms(boxes, pos_scores.squeeze(), iou_threshold=hyperparameters['iou_threshold'])
            all_predictions.extend(pos_scores[keep].cpu().numpy())
            all_ground_truths.extend(labels[keep].cpu().numpy())

            # Calculating validation metric stuff ...

            predictions = (scores.squeeze() > 0.5).int()
            # predictions = labels


            num_correct += (predictions == labels.squeeze()).sum().item()
            num_samples += len(predictions)
           
            # Save predictions and labels
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
 
 
            # Find misclassified examples
            # misclassified_mask = predictions != labels
            # if misclassified_mask.any():
            #    misclassified_images = images[misclassified_mask].cpu()
            #    misclassified_labels = labels[misclassified_mask].cpu().numpy()
            #    misclassified_predictions = predictions[misclassified_mask].cpu().numpy()
 
                # Append only misclassified examples
            #    misclassified.extend(
            #        zip(misclassified_images, misclassified_labels, misclassified_predictions))



    avg_precision = AveragePrecision(task='binary')


    # cast all_ground_truths to int tensor

    all_predictions = torch.tensor(np.array(all_predictions).flatten()).float()
    all_ground_truths = torch.tensor(np.array(all_ground_truths).flatten()).int()
    avg_precision.update(torch.tensor(all_predictions), all_ground_truths)
    results = avg_precision.compute()
    print(f"Average Precision: {results}")
    accuracy = float(num_correct)/float(num_samples)
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy * 100:.3f}%")
    classes = ('Background', 'Positive') # TODO: CHECK IF THIS IS CORRECT

 
    # Create confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.set_theme(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix', fontsize=20, fontweight='bold', color='red')
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
 
    # save_misclassified_images(misclassified, save_dir)
 
    model.train()
    return accuracy, results
 
def save_misclassified_images(misclassified, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    for idx, (images, labels, predictions) in enumerate(misclassified):
        # Save only when true and predicted labels are different
        if labels != predictions:
            image = rescale_0_1(images)
            image = TF.to_pil_image(image)
            # Naming format: misclassified_<index>_true_<true label>_predicted_<predicted label>.png
            image_name = f"misclassified_{idx}_true_label_{labels}_predicted_label_{predictions}.png"
            image_path = os.path.join(save_dir, image_name)
            image.save(image_path)
 
def rescale_0_1(image):
    """Rescale pixel values to range [0, 1] for visualization purposes only."""
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image-min_val)/abs(max_val-min_val)
    return rescaled_image
 
 
 
################### MAIN CODE ###################

 
hyperparameters = {
    'step size': 30,
    'learning rate': 0.0001,
    'epochs': 100,
    'gamma': 0.9,
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
    'iou_threshold': 0.5
}

# hyperparameters = {
#     'step size': 1,
#     'learning rate': 0.01,
#     'epochs': 1,
#     'gamma': 0.9,
#     'momentum': 0.9,
#     'optimizer': 'Adam',
#     'number of classes': 1,
#     'device': 'cuda',
#     'image size': 256,
#     'backbone': 'mobilenet_v3_large', # "mobilenet_v3_large" or "resnet152"
#     'torch home': 'TorchvisionModels',
#     'network name': 'Test-0',
#     'beta1': 0.9,
#     'beta2': 0.999,
#     'epsilon': 1e-08,
#     'number of workers': 3,
#     'weight decay': 0.0005,
#     'scheduler': 'Yes',
#     'iou_threshold': 0.5
# }
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet152', help='The backbone of the model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()
    hyperparameters['backbone'] = args.backbone
    hyperparameters['epochs'] = args.epochs

    print(f"Hyperparameters: {hyperparameters}")

     
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



    model = MultiModel(backbone=hyperparameters['backbone'], hyperparameters=hyperparameters, load_pretrained=True).to(device)
    # summary(model, (3, hyperparameters['image size'], hyperparameters['image size']))
    # model.count_parameters()
    
    run_dir = "Results"
    os.makedirs(run_dir, exist_ok=True)
    modeltype = hyperparameters['backbone']
    epochs = hyperparameters['epochs']
    modeltype_directory = os.path.join(run_dir, f'{modeltype}_{epochs}')
    os.makedirs(modeltype_directory, exist_ok=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for training")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for validation")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # !! Do not shuffle here and do not change batch_size !!
    print("Created a new Dataloader for testing")
    
    log_dir = os.path.join(modeltype_directory, f'{hyperparameters["network name"]}_{hyperparameters["optimizer"]}_Scheduler_{hyperparameters["scheduler"]}')
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)
    
    accuracy = train_net(model, logger, hyperparameters, device,
                                loss_function, train_loader, val_loader, test_loader, modeltype_directory)
    print(f"Final accuracy: {accuracy}")
    
    