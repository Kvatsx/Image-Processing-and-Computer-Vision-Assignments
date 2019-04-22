
# coding: utf-8

# In[139]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy as sp
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve 


# In[32]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[33]:


train_df.head()


# In[34]:


test_df.head()


# In[35]:


print(len(train_df.columns))
print(len(test_df.columns))


# In[36]:


trainY = train_df['Creditability']
testY = test_df['Creditability']
train_df.drop(train_df.columns[0], axis=1,inplace=True)
test_df.drop(test_df.columns[0],axis=1,inplace=True)


# In[37]:


train_df.drop(train_df.columns[0], axis=1,inplace=True)
test_df.drop(test_df.columns[0],axis=1,inplace=True)


# In[38]:


train_df.head()
train_df.sample(frac=1)


# In[39]:


test_df.head()


# In[40]:


trainX = np.array(train_df)
testX = np.array(test_df)
# clf = DecisionTreeClassifier(random_state=0)
print(trainX)
print(testX)


# In[295]:


min_leaf = [8,20,30,45,75,90,100]
min_split = [45,75,90,100]
depth = [3,5,8,10,20]
criterion = ['gini','entropy']
params = {'criterion':criterion, 'max_depth':depth,'min_samples_split':min_split,'min_samples_leaf':min_leaf}
# params = {'max_depth':depth,'min_samples_leaf':min_leaf}
grid = GridSearchCV(DecisionTreeClassifier(),param_grid = params,cv=5)
grid.fit(trainX,trainY)
bestparams = grid.best_params_
clf = DecisionTreeClassifier(max_depth=bestparams['max_depth'],
                             min_samples_split=bestparams['min_samples_split'],
                             min_samples_leaf = bestparams['min_samples_leaf'],
                             criterion = bestparams['criterion'])
clf = DecisionTreeClassifier(max_depth=8,min_samples_split=45,min_samples_leaf=8)
clf.fit(trainX,trainY)
print(clf.score(trainX,trainY))
print(clf.score(testX,testY))


# In[134]:


print(grid.best_params_)


# In[296]:


pickle_out = open('grid_result_dt.pickle','wb')
pickle.dump(grid,pickle_out)
pickle_out.close()
pickle_out = open('clf_dt.pickle','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[ ]:


min_leaf = [8,20,30,45,75,90,100]
min_split = [45,75,90,100]
depth = [3,5,8,10,20]
estimators = [25,40,50,70,90]
criterion = ['gini','entropy']
params = {'n_estimators':estimators,'max_depth':depth,'min_samples_split':min_split,'min_samples_leaf':min_leaf,'criterion':criterion}
grid = GridSearchCV(RandomForestClassifier(),param_grid = params,cv=5,verbose=1)
grid.fit(trainX,trainY)
bestparams = grid.best_params_
# clf = RandomForestClassifier(max_depth=10,min_samples_split=30,min_samples_leaf=5,n_estimators=50)
clf = RandomForestClassifier(max_depth=bestparams['max_depth'],
                             min_samples_split=bestparams['min_samples_split'],
                             min_samples_leaf = bestparams['min_samples_leaf'],
                             n_estimators = bestparams['n_estimators'],
                             criterion = bestparams['criterion'])
clf.fit(trainX,trainY)
print(clf.score(trainX,trainY))
print(clf.score(testX,testY))


# In[131]:


print(grid.best_params_)


# ## Parts b) and c)

# In[135]:


pickle_out = open('grid_result_dt.pickle','wb')
pickle.dump(grid,pickle_out)
pickle_out.close()
pickle_out = open('clf_dt.pickle','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[132]:


pickle_out = open('grid_result_rf.pickle','wb')
pickle.dump(grid,pickle_out)
pickle_out.close()
pickle_out = open('clf_rf.pickle','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[136]:


pickle_in = open('grid_result_dt.pickle','rb')
grid1 = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('clf_dt.pickle','rb')
clf1 = pickle.load(pickle_in)
print(clf1.score(testX,testY))
pickle_in.close()


# In[137]:


pickle_in = open('grid_result_rf.pickle','rb')
grid2 = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('clf_rf.pickle','rb')
clf2 = pickle.load(pickle_in)
print(clf2.score(testX,testY))
pickle_in.close()


# In[275]:


train_sizes, train_scores, test_scores = learning_curve(clf, trainX, trainY, cv=3, train_sizes=np.linspace(0.1,1,5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes,train_scores_mean,'o-')
plt.plot(train_sizes,test_scores_mean,'o-')
plt.savefig('lc1.png')
plt.show()


# In[266]:


train_sizes, train_scores, test_scores = learning_curve(clf2, trainX, trainY, cv=3, train_sizes=np.linspace(0.1,1,5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes,train_scores_mean,'o-')
plt.plot(train_sizes,test_scores_mean,'o-')
plt.savefig('lc2.png')
plt.show()


# In[219]:


dt_results = pd.DataFrame(grid1.cv_results_)
print(dt_results.head())
print(dt_results.columns)


# In[149]:


rf_results = pd.DataFrame(grid2.cv_results_)
print(dt_results.head())
print(dt_results.columns)


# In[277]:


dt_results.to_csv('decisiontreeresults.csv')
rf_results.to_csv('randomforestclassifier.csv')


# ## d) Calculating variance of error

# In[292]:


bestresults1 = dt_results[dt_results['params']==grid1.best_params_]
bestresults2 = rf_results[rf_results['params']==grid2.best_params_]
bestresults1 = bestresults1.loc[:, 'split0_test_score':'split2_train_score':2]
bestresults2 = bestresults2.loc[:, 'split0_test_score':'split2_train_score':2]
bestresults1 = np.array(bestresults1)
bestresults2 = np.array(bestresults2)
print(bestresults1)
print(bestresults2)
print(np.var(bestresults1))
print(np.var(bestresults2))
# print(bestresults1)
# print(bestresults2)
# print(bestresults1)
# print(bestresults2)

