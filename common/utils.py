'''Some helper functions for PyTorch
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from enum import Enum
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class DatasetName(Enum):
    CIFAR10 = 1

class CutOut:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)
        mask[y1:y2, x1:x2] = 0
        img *= mask[:, :, np.newaxis]
        return torch.from_numpy(img)

def load_transformed_dataset(dataset_name, batch_size):
    if dataset_name == DatasetName.CIFAR10:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutOut(16),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader, classes

    raise ValueError("Invalid dataset name: Only 'CIFAR10' is supported for now")

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# Function to plot training metrics (loss and accuracy)
def plot_train_metrics(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

# Function to plot a sample of data from the data loader
def plot_sample_data(data_loader):
    batch_data, batch_label = next(iter(data_loader))
    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_misclassified_images_from_model(model, device, data_loader, class_labels, image_count):
  correct = 0
  figure = plt.figure(figsize=(15,15))
  count = 0
  with torch.no_grad():
      for data, target in data_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

          for idx in range(len(pred)):
            i_pred, i_act = pred[idx], target[idx]
            if i_pred != i_act:
                annotation = "Actual: %s, Predicted: %s" % (class_labels[i_act], class_labels[i_pred])
                count += 1
                plt.subplot(5, 2, count)
                plt.axis('off')
                imshow(data[idx].cpu())
                plt.annotate(annotation, xy=(0,0), xytext=(0,-1.2), fontsize=13)
            if count == image_count:
                return

def show_gradcam_on_misclassified_images_from_model(model, device, target_layer, data_loader, class_labels, image_count):
  correct = 0
  figure = plt.figure(figsize=(15,15))
  count = 0
  gradcam = GradCAM(model=model, target_layers=[target_layer])
  unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
  ])
  targets = None
  for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      cam = gradcam(input_tensor=data, targets=targets)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

      for idx in range(len(pred)):
        i_pred, i_act = pred[idx], target[idx]
        if i_pred != i_act:
            annotation = "Actual: %s, Predicted: %s" % (class_labels[i_act], class_labels[i_pred])
            count += 1
            unnormalized_image = unnormalize(data[idx])
            unnormalized_image = np.clip(unnormalized_image, 0, 1)  # Clip to range [0, 1]
            cam_image = show_cam_on_image(unnormalized_image, cam[idx], use_rgb=True)
            plt.subplot(5, 2, count)
            plt.axis('off')
            imshow(cam_image)
            plt.annotate(annotation, xy=(0,0), xytext=(0,-1.2), fontsize=13)
        if count == image_count:
            return
