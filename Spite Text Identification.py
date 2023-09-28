#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib as pyplot


# In[3]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 


# In[4]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[7]:


dataset= pd.read_csv(r"labeled_data.csv")


# In[8]:


dataset.head()


# In[9]:


dataset.tail()


# In[10]:


dataset.describe()


# In[11]:


dataset.info()


# In[12]:


dt_transformed = dataset[['class','tweet']]
y=dt_transformed.iloc[:,:-1].values
print(y)


# In[13]:


newfeatures = dataset.drop(['class'],axis=1)  
newlabel = dataset['class']


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.countplot(x=newlabel)
ax.set(xlabel='class')
plt.show()
plt.close()


# In[15]:


ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
y=np.array(ct.fit_transform(y))


# In[16]:


y_df=pd.DataFrame(y)
y_hate=np.array(y_df[0])
y_offensive =np.array(y_df[1])


# In[17]:


fig = plt.figure(figsize=(10,5))
sns.heatmap(dataset.corr(),annot=True)
plt.show() 
plt.close()


# In[18]:


corpus=[]
for i in range(0,24783):
  review = re.sub('[^a-zA-Z]',' ',dt_transformed['tweet'][i])
  review = review.lower()
  review= review.split()
  ps=PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[19]:


cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(corpus).toarray()


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, y_hate, test_size=0.30, random_state=0)


# In[21]:


#KNN
classifier_KNN = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier_KNN.fit(x_train,y_train)


# In[23]:


#LR
classifier_lr =LogisticRegression(random_state =0)
classifier_lr.fit(x_train, y_train)


# In[24]:


#SVM
classifier_svm = svm.SVC()
classifier_svm.fit(x_train, y_train)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators=10, random_state =0)
classifier_RF.fit(x_train, y_train)


# In[ ]:


y_pred_KNN = classifier_KNN.predict(x_test)
cm3 = confusion_matrix(y_test,y_pred_KNN)
print('Confusion Matrix for KNN:\n')
print(cm3)


# In[27]:


y_pred_lr = classifier_lr.predict(x_test)
cm2=confusion_matrix(y_test,y_pred_lr)
print('Confusion Matrix for Logistic Regression:\n')
print(cm2)


# In[28]:


#SVM
y_pred_svm =classifier_svm.predict(x_test)
cm1=confusion_matrix(y_test,y_pred_svm)
print('Confusion Matrix for SVM:\n')
print(cm1)


# In[29]:


y_pred_RF = classifier_RF.predict(x_test)
cm4 = confusion_matrix(y_test, y_pred_RF)
print('Confusion Matrix for Random Forest:\n')
print(cm4)


# In[30]:


svm_score = accuracy_score(y_test, y_pred_svm)
lr_score  = accuracy_score(y_test, y_pred_lr)
KNN_score = accuracy_score(y_test, y_pred_KNN)
RF_score  = accuracy_score(y_test, y_pred_RF)


# In[31]:


print('Logistic regresion     :',str(lr_score))
print('Support Vector Machine :', str(svm_score))
print('K-Nearest Neighbor     :', str(KNN_score))
print('Random Forest          :', str(RF_score))

