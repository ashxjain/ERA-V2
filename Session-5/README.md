# MNIST Image Classification with PyTorch

This repository contains code for training a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch.

## Files

### model.py

The `model.py` file defines the architecture of the convolutional neural network (CNN) used for image classification. It contains the `Net` class, which specifies the layers and operations of the neural network. Here's what each part of the `model.py` file does:

- **Importing Libraries**: The file starts by importing the necessary PyTorch libraries for defining neural networks (`torch`), neural network modules (`torch.nn`), and functional operations (`torch.nn.functional`).

- **Defining the Neural Network Architecture**: The `Net` class defines the structure of the CNN. It specifies the convolutional layers, pooling layers, and fully connected layers of the network. Each layer is defined using PyTorch's neural network module classes (`nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear`, etc.).

- **Forward Pass**: The `forward` method of the `Net` class specifies how input data is processed through the network layers to generate output predictions. It applies activation functions (`F.relu`) to the output of convolutional layers and reshapes the output tensor before passing it to fully connected layers.

### utils.py

The `utils.py` file contains utility functions used for training and testing the CNN model. These functions facilitate data loading, training, testing, and accuracy calculation. Here's what each part of the `utils.py` file does:

- **Importing Libraries**: Similar to `model.py`, the file starts by importing necessary libraries such as PyTorch, torchvision, matplotlib, and tqdm.

- **Defining Utility Functions**: The file contains several utility functions:
  - `train`: A function to train the CNN model using the training dataset. It iterates over batches of data, computes predictions, calculates loss, performs backpropagation, and updates model parameters.
  - `test`: A function to evaluate the trained model using the test dataset. It computes predictions, calculates loss, and calculates accuracy.
  - `plot_train_metrics`: A function to plot the training & testing losses and accuracies of the model. This will give us a good idea of how our model is performing.
  - `plot_sample_data`: A function to plot sample data from the data loader. The data loader can be test or train data. It will plot some sample images. This will give us a good idea of our dataset looks like.
  - Other utility functions may be added depending on the specific requirements of the project.

### S5.ipynb

The `S5.ipynb` Jupyter notebook serves as the main script for training and evaluating the CNN model. It imports the `model.py` and `utils.py` files to define the model architecture and perform training and testing. Here's what each part of the `S5.ipynb` notebook does:

- **Importing Libraries and Dependencies**: The notebook starts by importing necessary libraries and dependencies, including PyTorch, torchvision, matplotlib, tqdm, and torchsummary.

- **Defining Data Loaders and Data Visualization**: Data loaders for the MNIST dataset are created using torchvision. This data is transformed before passing to the model. Also, the data is visualized to get a sense of how the data looks like.

- **Defining Neural Network Model**: Network model is important and loaded into GPU. Model summary is seen to get the information on the number of parameters and how the model looks like.

- **Defining Hyperparameters**: Hyperparameters such as batch size, learning rate, and number of epochs are defined.

- **Training and Evaluating the Model**: The notebook contains code cells for training the CNN model using the `train` function from `utils.py` and evaluating its performance using the `test` function. It iterates over multiple epochs, updating the model parameters and monitoring training progress.

- **Plotting Training Metrics**: After training, the notebook plots training loss, training accuracy, test loss, and test accuracy to visualize the performance of the model over epochs.

