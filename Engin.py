#!/usr/bin/env python
# coding: utf-8

# # Sample ML Model of Recommendation Engine                         
# scroll to very bottom to see the demo

# In[40]:


#importing neccesary packages/modules
import warnings
warnings.simplefilter("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier



# In[41]:


#Reads fake_student_data.csv which is a mockup of info of student profiles and their interests. 
#To build a real model like this you would first need to collect this data from students. Qualtrics is a good option
#for a free survey survice that automatically generates a csv data file

data = pd.read_csv('fake_student_data.csv', skipinitialspace = True)
print("Every Feature: \n", data.columns)


# In[42]:


#Perform one hot encoding on all features with categorical data to make every column in the dataset numerical
data = pd.get_dummies(data, columns = ['gender','education_level', 'career_fields_of_interest'], drop_first = False)
data = data.drop(columns=['id'])
print("Columns After One Hot Encoding")
print(data.columns)


# In[43]:


#show number of missing/invalid data
#fills in missing values with the average of their column
data = data.fillna(data.mean())
#clean_data.to_csv('clean_data.csv')


# In[44]:


corr_matrix = data.corr()
corr_list = []
print("Correlation of Features with a value greater than 0.6 or less than -0.6")
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.6: #print out high correlation
            corr_list.append([corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]])

corr_frame = pd.DataFrame(corr_list, columns = ['Feature 1', 'Feature 2', 'Correlation'])
corr_frame = corr_frame.style.set_precision(4)


# In[45]:


'''
# Normalize numerical variables
scaler = MinMaxScaler()
numerical_cols = ['age', 'interest_in_connections', 'interest_in_creations',
                  'interest_in_exploration', 'interest_in_health', 'interest_in_living',
                  'interest_in_justice']
clean_data[numerical_cols] = scaler.fit_transform(clean_data[numerical_cols])
# Create feature vectors for each student
features = pd.get_dummies(clean_data).values
'''


# # The Dataset
# This is a mockup of what a survey to students would look like. I assume questions over age, interests, experiences would be asked. Student answers can be used to match students with relevent resources. Here I randomly generated student profiles and assigned them 3 resources (randomly). These 3 resources represent recommendations we would make from what we know of them.
# 
# Let's say we ask existing students to tell us what their 3 favorite resources are. When a new student comes, we can use their information and reference our existing student data (for current students) to suggest resources that other students (similar to the new student) enjoyed. 

# In[46]:


r1 = data.loc[:,['resource1']]
r2 = data.loc[:,['resource2']]
r3 = data.loc[:,['resource3']]
features_data = data.drop(columns=['resource1', 'resource2', 'resource3'])
features_data


# # The Models
# Because the data is randomly generated (there's no real correlations between the student attributes and which resources they're tagged with, most models will produce 50% correctness at most.The following models use information about the user to predict resources the user will enjoy (from information of other students).

# In[47]:


scaler = StandardScaler()
neighbor = KNeighborsClassifier(n_neighbors = 3)
pip = Pipeline(steps = [('scaler',scaler),('knn',neighbor)])
param_grid = {
    'knn__n_neighbors': list(range(1, 3))
}
grid = GridSearchCV(pip, param_grid, cv = 5, scoring = 'accuracy')
knn_pred = cross_val_predict(grid, features_data, r1, cv = 5)
print("Accuracy: \n", accuracy_score(r1, knn_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r1, knn_pred))
print()
print("Classification Report: \n", classification_report(r1, knn_pred))


pip = Pipeline(steps = [('scaler',scaler),('knn',neighbor)])
grid = GridSearchCV(pip, param_grid, cv = 5,  scoring = 'accuracy')
knn_pred = cross_val_predict(grid, features_data, r2, cv = 5)
print("Accuracy: \n", accuracy_score(r2, knn_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r2, knn_pred))
print()
print("Classification Report: \n", classification_report(r2, knn_pred))

grid = GridSearchCV(pip, param_grid, cv = 5,  scoring = 'accuracy')
knn_pred = cross_val_predict(grid, features_data, r3, cv = 5)
print("Accuracy: \n", accuracy_score(r3, knn_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r3, knn_pred))
print()
print("Classification Report: \n", classification_report(r3, knn_pred))


# In[48]:


import imblearn
from imblearn.over_sampling import SMOTE

x_train, x_test, y_train, y_test = train_test_split(features_data, r1, train_size = 0.8)
print("\nShapes of Test/Train Sets")
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

tree_classifier = DecisionTreeClassifier(criterion = 'entropy')
print("after smote distribution is now ", y_train['resource1'].value_counts())
tree_classifier.fit(x_train, y_train)

x_train, x_test, y_train, y_test = train_test_split(features_data, r2, train_size = 0.8)
print("\nShapes of Test/Train Sets")
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print("after smote distribution is now ", y_train['resource2'].value_counts())
tree_classifier.fit(x_train, y_train)
prediction = tree_classifier.predict(x_test)


x_train, x_test, y_train, y_test = train_test_split(features_data, r3, train_size = 0.8)
print("\nShapes of Test/Train Sets")
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print("after smote distribution is now ", y_train['resource3'].value_counts())
tree_classifier.fit(x_train, y_train)
prediction = tree_classifier.predict(x_test)


# In[49]:


bayes_classifier = GaussianNB()
pip = Pipeline(steps = [('bayes',bayes_classifier)])
print("r1")
naive_cross = cross_val_score(pip, features_data, r1, cv = 5)
print("Accuracy: ", naive_cross.mean() * 100)
naive_pred = cross_val_predict(pip, features_data, r1, cv = 5)
print("Confusion Matrix: \n", confusion_matrix(r1, naive_pred))
print()
print("Classification Report: \n", classification_report(r1, naive_pred))

print("r2")
naive_cross = cross_val_score(pip, features_data, r2, cv = 5)
print("Accuracy: ", naive_cross.mean() * 100)
naive_pred = cross_val_predict(pip, features_data, r3, cv = 5)
print("Confusion Matrix: \n", confusion_matrix(r1, naive_pred))
print()
print("Classification Report: \n", classification_report(r1, naive_pred))


print("r3")
naive_cross = cross_val_score(pip, features_data, r3, cv = 5)
print("Accuracy: ", naive_cross.mean() * 100)
naive_pred = cross_val_predict(pip, features_data, r3, cv = 5)
print("Confusion Matrix: \n", confusion_matrix(r1, naive_pred))
print()
print("Classification Report: \n", classification_report(r1, naive_pred))


# In[50]:


mlp = MLPClassifier()
pip = Pipeline(steps = [('scaler', scaler), ('mlp', mlp)])
param_grid = {
    'mlp__activation': ['logistic', 'tanh', 'relu'],
    'mlp__hidden_layer_sizes' :[(5,),(10,),(15,)]
}


grid = GridSearchCV(pip, param_grid, cv = 2, scoring = 'accuracy')
mlp_pred = cross_val_predict(grid, features_data, r1, cv = 2)
print("Accuracy: \n", accuracy_score(r1, mlp_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r1, mlp_pred))
print()
print("Classification Report: \n", classification_report(r1, mlp_pred))

grid = GridSearchCV(pip, param_grid, cv = 2, scoring = 'accuracy')
mlp_pred = cross_val_predict(grid, features_data, r2, cv = 2)
print("Accuracy: \n", accuracy_score(r1, mlp_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r2, mlp_pred))
print()
print("Classification Report: \n", classification_report(r2, mlp_pred))

grid = GridSearchCV(pip, param_grid, cv = 2, scoring = 'accuracy')
mlp_pred = cross_val_predict(grid, features_data, r3, cv = 2)
print("Accuracy: \n", accuracy_score(r1, mlp_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r3, mlp_pred))
print()
print("Classification Report: \n", classification_report(r3, mlp_pred))


# In[12]:


from sklearn.ensemble import AdaBoostClassifier
tree_base_classifier =  DecisionTreeClassifier(criterion = 'entropy')

adc = AdaBoostClassifier(base_estimator = tree_base_classifier, n_estimators = 5)
pip = Pipeline(steps = [('ada',adc)])
ada_pred = cross_val_predict(pip, features_data, r1, cv = 5)
print("Accuracy: \n", accuracy_score(r1, ada_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r1, ada_pred))
print()
print("Classification Report: \n", classification_report(r1, ada_pred))


ada_pred = cross_val_predict(pip, features_data, r2, cv = 5)
print("Accuracy: \n", accuracy_score(r2, ada_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r2, ada_pred))
print()
print("Classification Report: \n", classification_report(r2, ada_pred))


ada_pred = cross_val_predict(pip, features_data, r3, cv = 5)
print("Accuracy: \n", accuracy_score(r3, ada_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(r3, ada_pred))
print()
print("Classification Report: \n", classification_report(r3, ada_pred))


# # Content Based Recommendations
# Let's make recommendations for students based off lessons's they've already completed
# https://helpseotools.com/text-tools/remove-special-characters
# https://helpseotools.com/text-tools/remove-special-characters
# Use the above to remove special characters and format text to be analyzed

# In[13]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

resources = pd.read_csv('resourcesText.csv', skipinitialspace = True)
#resources = resources.drop(columns=['Unit','Category', 'Subject', 'Level'])
#resources = pd.get_dummies(resources, columns = ['Category','Subject'], drop_first = False)
resources = resources.fillna(data.mean())
resources


# In[31]:


tfidf = TfidfVectorizer(stop_words='english')
resources['verview'] = resources['Overview'].fillna('')
overview_matrix = tfidf.fit_transform(resources['Overview'])
similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
mapping = pd.Series(resources.index,index = resources['Unit'])


# In[32]:


def recommend_resources(input):
    index = mapping[input]
    similarity_score = list(enumerate(similarity_matrix[index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:4]
    indices = [i[0] for i in similarity_score]
    return (resources['Unit'].iloc[indices])


# In[33]:


def predict_for_student (input):
    x_train, x_test, y_train, y_test = train_test_split(features_data, r1, train_size = 0.8)

    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(x_train, y_train)
    prediction1 = tree_classifier.predict(input)

    x_train, x_test, y_train, y_test = train_test_split(features_data, r2, train_size = 0.8)
    tree_classifier.fit(x_train, y_train)
    prediction2 = tree_classifier.predict(input)


    x_train, x_test, y_train, y_test = train_test_split(features_data, r3, train_size = 0.8)
    tree_classifier.fit(x_train, y_train)
    prediction3 = tree_classifier.predict(input)
    
    return prediction1, prediction2, prediction3


# # DEMO

# In[34]:


prediction = recommend_resources('Unit 1')
print('Based on what lessons you like we recommend ')
print(prediction)


# In[39]:


test_student = data.iloc[0]
actual1, actual2, actual3 = test_student['resource1'], test_student['resource2'], test_student['resource3']
test_student = test_student.drop(['resource1', 'resource2', 'resource3'], axis=0)
test_student = test_student.to_frame().T
prediction1, prediction2, prediction3 = predict_for_student(test_student)
print('Based on your information we recommend ', prediction1, prediction2, 'and', prediction3)
print('You actually said you would enjoy ', actual1, actual2, 'and', actual3)

