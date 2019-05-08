'''
 @Author: Kaustav Vats 
 @Roll-Number: 2016048 
 @Date: 2019-04-22 23:28:19 
'''
# Ref:- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py

import numpy as np
import cv2
from utils import ReadData
import torch
import torch.nn as nn
import pickle
import torchvision
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


PATH = "./Data/"

'''---------------------Reading dataset----------------------''' 
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# trainset = torchvision.datasets.CIFAR10(root='./Data/Q2', train=True, download=True, transform=trans)
# testset = torchvision.datasets.CIFAR10(root='./Data/Q2', train=False, download=True, transform=trans)

# train_loader = torch.utils.data.DataLoader(trainset, shuffle=True)
# test_loader = torch.utils.data.DataLoader(testset, shuffle=False)

print("[+] Data Loader ready")

'''---------------------Features Extraction------------------'''

# model = models.alexnet(pretrained=True)

# def ExtractFeatures(loader, pic):
#     # features = []
#     Labels = []
#     for i, (images, labels) in enumerate(loader):
#         out = model(images)
#         pickle.dump(out, pic)   
#         # features.append(out)
#         Labels.append(labels)
    
#     # features = np.asarray(features)
#     Labels = np.asarray(Labels)
#     return Labels

# pickle_out = open(PATH + 'train.pickle','wb')
# Y_Train = ExtractFeatures(train_loader, pickle_out)
# pickle_out.close()

# pickle_out = open(PATH + 'test.pickle','wb')
# Y_Test = ExtractFeatures(test_loader, pickle_out)
# pickle_out.close()

# np.save(PATH + "X_Train.npy", X_Train)
# np.save(PATH + "Y_Train.npy", Y_Train)
# np.save(PATH + "X_Test.npy", X_Test)
# np.save(PATH + "Y_Test.npy", Y_Test)

# # X_Train = np.load(PATH + "X_Train.npy")
Y_Train = np.load(PATH + "Y_Train.npy")
# # X_Test = np.load(PATH + "X_Test.npy")
Y_Test = np.load(PATH + "Y_Test.npy")
# # print("Feature Extraction Done")

# def LoadPickledData():
#     Train = []
#     Test = []
#     pickle_out = open(PATH + 'train.pickle','rb')
#     for i in range(Y_Train.shape[0]):
#         feature = pickle.load(pickle_out)
#         temp = feature.detach().numpy()
#         temp = np.reshape(temp, (temp.shape[1], ))
#         Train.append(temp)
#     pickle_out.close()

#     pickle_out = open(PATH + 'test.pickle', 'rb')
#     for i in range(Y_Test.shape[0]):
#         feature = pickle.load(pickle_out)
#         temp = feature.detach().numpy()
#         temp = np.reshape(temp, (temp.shape[1], ))
#         Test.append(temp)
#     pickle_out.close()

#     Train = np.asarray(Train)
#     Test = np.asarray(Test)
#     print("Train.shape {}".format(Train.shape))
#     print("Test.shape {}".format(Test.shape))
#     return Train, Test

# Train, Test = LoadPickledData()

# np.save(PATH + "Train.npy", Train)
# np.save(PATH + "Test.npy", Test)

Train = np.load(PATH + "Train.npy")
Test = np.load(PATH + "Test.npy")
print("Done")

'''---------------------Testing------------------------------'''

# clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
# clf.fit(Train, Y_Train)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
clf.fit(Train, Y_Train)


Acc = clf.score(Train, Y_Train)
print("[RFC] Accuracy Train: {}%".format(Acc*100))
Acc = clf.score(Test, Y_Test)
print("[RFC] Accuracy Test: {}%".format(Acc*100))
pred = clf.predict(Test)

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

ConfusionMatrix(Y_Test, pred, "asd")