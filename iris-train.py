# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim

# Load data
iris = load_iris()
data = iris.data
labels = iris.target

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2)

# Convert to tensors
train_x = torch.tensor(train_data, dtype=torch.float32)
test_x = torch.tensor(test_data, dtype=torch.float32)
train_y = torch.LongTensor(train_labels)
test_y = torch.LongTensor(test_labels)

# Create dataset
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

# Create dataloader
train_batch = DataLoader(
        dataset=train_dataset,
        batch_size=5,
        shuffle=True,
        # num_workers=2
        )
test_batch = DataLoader(
        dataset=test_dataset,
        batch_size=5,
        shuffle=True,
        # num_workers=2
        )


# Create model
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Definition of hyperparameters
D_in = 4
H = 100
D_out = 3
epochs = 100

# Load model
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
net = Net(D_in, H, D_out).to(device)

print("Device: {}".format(device))

# Definition of loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Create list to store loss and accuracy
train_loss_list: list = []
train_accuracy_list: list = []
test_loss_list: list = []
test_accuracy_list: list = []

# Train model
for epoch in range(epochs):
    print("Epoch: {}".format(epoch+1))
    train_loss = 0.0
    train_accuracy = 0.0
    test_loss = 0.0
    test_accuracy = 0.0

    # Train
    net.train()
    for data, labels in train_batch:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred_prob = net(data)
        loss = criterion(y_pred_prob, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(y_pred_prob.data, 1)
        train_accuracy += (predicted == labels).sum().item() / len(labels)

    # Calculate average loss and accuracy
    batch_train_loss = train_loss / len(train_batch)
    batch_train_accuracy = train_accuracy / len(train_batch)

    # Test
    net.eval()
    with torch.no_grad():
        for data, labels in test_batch:
            data = data.to(device)
            labels = labels.to(device)
            y_pred_prob = net(data)
            loss = criterion(y_pred_prob, labels)
            test_loss += loss.item()
            _, predicted = torch.max(y_pred_prob.data, 1)
            test_accuracy += (predicted == labels).sum().item() / len(labels)

    # Calculate average loss and accuracy
    batch_test_loss = test_loss / len(test_batch)
    batch_test_accuracy = test_accuracy / len(test_batch)

    # Show loss and accuracy by epoch
    print("Train loss: {:.3f}, Train accuracy: {:.3f}"
          .format(batch_train_loss, batch_train_accuracy))
    print("Test loss: {:.3f}, Test accuracy: {:.3f}"
          .format(batch_test_loss, batch_test_accuracy))

    # Store loss and accuracy
    train_loss_list.append(batch_train_loss)
    train_accuracy_list.append(batch_train_accuracy)
    test_loss_list.append(batch_test_loss)
    test_accuracy_list.append(batch_test_accuracy)

# Plot loss
plt.figure()
plt.plot(train_loss_list, label='train')
plt.plot(test_loss_list, label='test')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot accuracy
plt.figure()
plt.plot(train_accuracy_list, label='train')
plt.plot(test_accuracy_list, label='test')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Show plot
plt.show()

# Save model
# torch.save(net.state_dict(), 'iris.pt')
