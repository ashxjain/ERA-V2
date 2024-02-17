import torch
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt

# Function to calculate the number of correct predictions in a batch
def get_correct_pred_count(pred, target):
    return pred.argmax(dim=1).eq(target).sum().item()

# Function for training the model
def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    model.train()  # Set the model to training mode
    pbar = tqdm(train_loader)  # Create a progress bar

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # Move data to device
        optimizer.zero_grad()  # Zero the gradients

        pred = model(data)  # Forward pass

        loss = criterion(pred, target)  # Calculate the loss
        train_loss += loss.item()  # Accumulate the loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Calculate accuracy
        correct += get_correct_pred_count(pred, target)
        processed += len(data)

        # Update progress bar description
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Save training metrics for plotting
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))

# Function for testing the model
def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()  # Set the model to evaluation mode

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # Move data to device

            output = model(data)  # Forward pass
            test_loss += criterion(output, target).item()  # Accumulate the loss

            # Calculate accuracy
            correct += get_correct_pred_count(output, target)

    # Calculate average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print test set performance
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

