'''
 @Author: Kaustav Vats 
 @Roll-Number: 2016048 
 @Date: 2019-04-22 23:28:19 
 @Ref:- [
    https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py
    ]
'''

import numpy as np
import cv2
from utils import ReadData, ReadLabels, FilterData, Turn2DataLoader
import pickle
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

PATH = "./Data/Q2-output/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))
print("Device Name:", torch.cuda.get_device_name(0))

'''---------------------Reading dataset----------------------''' 

# X_Train, Y_Train, X_Test, Y_Test = ReadData()
# Labels = ReadLabels(["automobile", "cat", "dog", "truck"])
# print("Labels Number: {}".format(Labels))

# X_Train, Y_Train = FilterData(X_Train, Y_Train, Labels)
# X_Test, Y_Test = FilterData(X_Test, Y_Test, Labels)

# print("Train Data shape: {}".format(X_Train.shape))
# print("Test Data shape: {}".format(X_Test.shape))

def Filter(data, classes):
    new_loader = []
    for (image, label) in data:
        if label == classes[0]:
            new_loader.append((image, 0))
        elif label == classes[1]:
            new_loader.append((image, 1))
        elif label == classes[2]:
            new_loader.append((image, 2))
        elif label == classes[3]:
            new_loader.append((image, 3))
    return new_loader

trans = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./Data/Q2', train=True, download=True, transform=trans)
testset = torchvision.datasets.CIFAR10(root='./Data/Q2', train=False, download=True, transform=trans)

trainset = Filter(trainset, [1, 3, 5, 9])
testset = Filter(testset, [1, 3, 5, 9])

'''------------------Hyper Parameters-----------------------'''

num_epochs = 30
batch_size = 512
learning_rate = 0.001

'''------------------Data Loader----------------------------'''
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

print("[+] Data Loader ready")

'''-------------------Convolution Neural Network-------------'''

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),            
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.Conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # self.Conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.Conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.Linear1 = nn.Linear(64 * 32 * 32, 500)
        # self.Linear2 = nn.Linear(500, 250)
        # self.Linear3 = nn.Linear(250, 4)
        # self.soft = nn.Softmax()
        self.Linear1 = nn.Sequential(
            nn.Linear(64 * 32 * 32, 500),
            nn.ReLU()  
        ) 
        self.Linear2 = nn.Sequential( 
            nn.Linear(500, 250),
            nn.ReLU()
        )
        self.Linear3 = nn.Sequential(
            nn.Linear(250, 4),
            nn.Softmax()
        ) 
        return

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.Linear1(out)
        out = self.Linear2(out)
        out = self.Linear3(out)
        return out

model = ConvNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Acc_list = []
Loss_list = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, total_loss/i))
    Loss_list.append(total_loss/i)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels).sum().item()
    Acc_list.append(100 * correct / total)  

model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Train Accuracy of the model on the train images: {} %'.format(100 * correct / total))
#     print('Train Loss of the model on the train images: {} %'.format(loss.item()))

Labels = []
pred= []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        Labels.extend(labels.cpu().numpy())
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        pred.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print('Test Loss of the model on the test images: {} %'.format(loss.item()))

torch.save(model.state_dict(), 'model.ckpt')


X = np.arange(num_epochs)
plt.figure()
plt.plot()
plt.plot(X, Acc_list, color='b', label="Accuracy")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs No of Epochs")
plt.legend(loc='best')
plt.savefig(PATH + "acc.png")

plt.figure()
plt.plot()
plt.plot(X, Loss_list, color='r', label="Loss")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs No of Epochs")
plt.legend(loc='best')
plt.savefig(PATH + "loss.png")



def ConfusionMatrix(actual, predicted, filename):
        classes = np.unique(predicted)
        # print("classes:", classes)
        cnf_matrix = confusion_matrix(actual, predicted)
        np.set_printoptions(precision=2)
        plt.figure()
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(PATH + "cm.png")
        # plt.show()
        plt.close()

ConfusionMatrix(Labels, pred, "asd")
