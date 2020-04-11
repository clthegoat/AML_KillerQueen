from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        x1 = x[:, :2, :, :]
        x2 = x[:, 2, :, :]
        x2 = torch.mean(x2, dim=(1, 2)).view(x.size(0), -1)
        x2[x2>-0.1] = 10
        x2[x2<-0.1] = -10
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = torch.cat((out, x2), dim=1)
        out = self.fc2(out)
        return out


class SleepingDataset(Dataset):
    def __init__(self, x, y):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.x = x
        self.y = y-1
        self.len = len(x)

    def __getitem__(self, index):
        img = torch.from_numpy(self.x[index])
        img = img.permute((2, 0, 1))
        return img, self.y[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    print()
    print('***************By Killer Queen***************')
    print('Loading data..')
    x_all = np.load('./data/train_x.npy')
    y_all = np.load('./data/train_labels.npy')

    num_epochs = 5
    num_classes = 3
    batch_size = 128
    learning_rate = 0.0007
    test_sub = 0

    x_all[:21600, :, :, 2] = (x_all[:21600, :, :, 2] - np.mean(x_all[:21600, :, :, 2])) / np.std(x_all[:21600, :, :, 2])
    x_all[21600:2*21600, :, :, 2] = (x_all[21600:2*21600, :, :, 2] - np.mean(x_all[21600:2*21600, :, :, 2])) / np.std(x_all[21600:2*21600, :, :, 2])
    x_all[:2*21600, :, :, 2] = (x_all[:2*21600, :, :, 2] - np.mean(x_all[:2*21600, :, :, 2] )) / np.std(x_all[:2*21600, :, :, 2] )

    if test_sub == 0:
        x_train = x_all[21600:]
        y_train = y_all[21600:]
        x_test = x_all[:21600]
        y_test = y_all[:21600]
    elif test_sub == 1:
        x_train = np.concatenate((x_all[21600:], x_all[2*21600:]), axis=0)
        y_train = np.concatenate((y_all[21600:], y_all[2*21600:]), axis=0)
        x_test = x_all[21600:2*21600]
        y_test = y_all[21600:2*21600]
    else:
        x_train = x_all[:2*21600]
        y_train = y_all[:2*21600]
        x_test = x_all[2*21600:]
        y_test = y_all[2*21600:]

    training_set = SleepingDataset(x_train, y_train)
    test_set = SleepingDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size,  shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model = Net(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print('Training..')
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    torch.save(model.state_dict(), './model.ckpt')
    # Test the model
    print('Evaluating..')
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    y_pred = np.zeros(len(x_test))
    y_true = np.zeros(len(x_test))
    cnt = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred[cnt:cnt+len(labels)] = predicted.data.cpu().numpy()
            y_true[cnt:cnt+len(labels)] = labels.data.cpu().numpy()
            cnt+=len(labels)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true, y_pred))
    print(balanced_accuracy_score(y_true, y_pred))

