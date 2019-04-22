import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import pickle
from sklearn.svm import SVC
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


pickle_in = open('./HW3_NN/data/Q2/train_CIFAR.pickle','rb')
traindata= pickle.load(pickle_in)
pickle_in = open('./HW3_NN/data/Q2/test_CIFAR.pickle','rb')
testdata= pickle.load(pickle_in)

trainX,trainY,testX,testY=traindata['X'],traindata['Y'],testdata['X'],testdata['Y']

# print(trainX.shape)
# print(trainY.shape)
# print(testX.shape)
# print(testY.shape)
#
# trainX = np.reshape(trainX,(trainX.shape[0],32,32,3))
# testX = np.reshape(testX,(testX.shape[0],32,32,3))
# trainimages = []
# testimages = []
# pickle_out = open('train_transformed_normalized.pickle','wb')
# for i in range(len(trainX)):
#     img = transforms.ToPILImage()(trainX[i])
#     img = img.resize((224,224))
#     img = transforms.ToTensor()(img)
#     transform_pipeline = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     img = transform_pipeline(img)
#     pickle.dump(img, pickle_out)
# pickle_out.close()
# pickle_out = open('test_transformed_normalized.pickle','wb')
# for i in range(len(testX)):
#     img = transforms.ToPILImage()(testX[i])
#     img = img.resize((224,224))
#     img = transforms.ToTensor()(img)
#     transform_pipeline = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     img = transform_pipeline(img)
#     pickle.dump(img, pickle_out)
# pickle_out.close()
#
#
# pickle_in = open('train_transformed_normalized.pickle','rb')
# alexnet = models.alexnet(pretrained=True)
# for i in tqdm(range(len(trainX))):
#     try:
#         img=pickle.load(pickle_in)
#         img = img.unsqueeze(0)
#         prediction = alexnet(img)
#         arr = prediction.data.numpy()
#         arr = np.reshape(arr,(arr.shape[1]))
#         trainimages.append(arr)
#     except EOFError:
#         break
# pickle_in = open('test_transformed_normalized.pickle','rb')
# alexnet = models.alexnet(pretrained=True)
# for i in tqdm(range(len(testX))):
#     try:
#         img=pickle.load(pickle_in)
#         img = img.unsqueeze(0)
#         prediction = alexnet(img)
#         arr = prediction.data.numpy()
#         arr = np.reshape(arr,(arr.shape[1]))
#         testimages.append(arr)
#     except EOFError:
#         break
#
# trainimages= np.array(trainimages)
# testimages = np.array(testimages)
# print(trainimages.shape)
# print(testimages.shape)
# pickle_out = open('trainimages_normalized.pickle','wb')
# pickle.dump(trainimages,pickle_out)
# pickle_out.close()
# pickle_out = open('testimages_normalized.pickle','wb')
# pickle.dump(testimages,pickle_out)
# pickle_out.close()





pickle_in = open('trainimages_normalized.pickle','rb')
trainimages = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('testimages_normalized.pickle','rb')
testimages = pickle.load(pickle_in)
pickle_in.close()

trainimages = scale(trainimages)
testimages = scale(testimages)
print(trainimages.shape)
print(testimages.shape)

C=[0.01,0.5,1,2]
params = {'C':C}
grid = GridSearchCV(SVC(kernel='linear'),params,cv=3)
grid.fit(trainimages,trainY)
print('grid search done!')
pickle_out = open('cifar_grid_norm','wb')
pickle.dump(grid,pickle_out)
pickle_out.close()
clf = SVC(kernel='linear', C=grid.best_params_['C'])
clf.fit(trainimages,trainY)
pickle_out = open('cifar10clf_norm_grid.pickle','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()
pickle_in = open('cifar10clf_norm_grid.pickle','rb')
clf = pickle.load(pickle_in)
pickle_in.close()


print("test accuracy")
print(clf.score(testimages,testY))
print("train accuracy")
print(clf.score(trainimages,trainY))
predictions = clf.predict(testimages)
print(confusion_matrix(testY,predictions))
decisions = clf.decision_function(testimages)
fpr,tpr,_ = roc_curve(testY,decisions)
plt.plot(fpr,tpr)
plt.savefig('roc_q2.png')
plt.show()
