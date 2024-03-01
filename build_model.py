import re
import time
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    TfidfTransformer,
    CountVectorize)
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    plot_precision_recall_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

''' Loading Dataset 1'''
train1 = pd.read_csv('train_1.csv')
print('Train Set 1 -----')
train1.head()
test1 = pd.read_csv('test_1.csv')
print('Test Set 1 -----')
test1.head()

''' Loading dataset 2'''
df = pd.read_csv('dataset2.csv')
print('Dataset 2 -----')
df.head()
df = df[['index','oh_label','Text']]
df.head()
''' Renaming Column names to follow a single naming convention '''
# Before renaming the Columns
("\nBefore modifying column names:\n", df.columns)

df.rename(columns = {'index':'id','oh_label':'label','Text':'tweet'}, inplace = True)
  
# After renaming the columns
print("\nAfter modifying first column:\n", df.columns)

'''Spliting the dataset 2 into training and testing dataset'''

train2, test2 = train_test_split(df, test_size=0.3, random_state=10, shuffle=True)

train2 = train2[['id', 'label', 'tweet']]
test2 = test2[['id', 'tweet']]

print('Train Set 2 -----')
train2.head()
print('Test Set 2 -----')
test2.head()

# Merging two Train Data Sets
train = pd.concat([train1, train2], ignore_index = True)
train.shape

train['label'].value_counts()

# Merging two Test Data Sets
test = pd.concat([test1,test2], ignore_index = True)
test.shape

'''Data Cleaning and upsampling to balance the class distribution'''
def  clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df
test_clean = clean_text(test, "tweet")
train_clean = clean_text(train, "tweet")

train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]
train_minority_upsampled = resample(train_minority, 
                                 replace=True,    
                                 n_samples=len(train_majority),   
                                 random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
train_upsampled['label'].value_counts()

train_upsampled.head()

'''Data Visualisation using WordCloud (use jupyter notebook for Visualisation)'''
fig, axs = plt.subplots(1,2 , figsize=(16,8))
text_pos = " ".join(train_clean['tweet'][train.label == 0])
text_neg = " ".join(train_clean['tweet'][train.label == 1])
train_cloud_pos = WordCloud(collocations = False, background_color = 'white').generate(text_pos)
train_cloud_neg = WordCloud(collocations = False, background_color = 'black').generate(text_neg)
axs[0].imshow(train_cloud_pos, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Non-Hate Comments')
axs[1].imshow(train_cloud_neg, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Hate Comments')

plt.show()

'''histogram to show class distribution before and after upsampling'''
plt.figure(figsize=(16,8))
sns.set_style('darkgrid')
sns.histplot(data = train['label'], color='black', legend=True)
sns.histplot(data = train_upsampled['label'], color = 'orange', legend=True)
plt.legend(['Initial_Data', 'Resampled_Data'])
plt.show()

print('--------------After Upsampling the Minority Class---------------')

fig, axs = plt.subplots(1,2 , figsize=(16,8))
text_pos = " ".join(train_upsampled['tweet'][train.label == 0])
text_neg = " ".join(train_upsampled['tweet'][train.label == 1])
train_cloud_pos = WordCloud(collocations = False, background_color = 'white').generate(text_pos)
train_cloud_neg = WordCloud(collocations = False, background_color = 'black').generate(text_neg)
axs[0].imshow(train_cloud_pos, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Non-Hate Comments')
axs[1].imshow(train_cloud_neg, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Hate Comments')

plt.show()

dt_trasformed = train_upsampled[['label', 'tweet']]
y = dt_trasformed.iloc[:, :-1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y))

y

print(y.shape)

y_df = pd.DataFrame(y)
y_hate = np.array(y_df[0])
y

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(train_upsampled['tweet']).toarray()
X

X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size = 0.30, random_state = 1)

# Using Random Forest

classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

# Start timer
start_time = time.time()

classifier_rf.fit(x_train, y_train)

# End timer
end_time = time.time()
# Calculate training time
training_time = end_time - start_time

print(f"Training time of Random Forest: {training_time:.6f} seconds")
# save the model to disk
filename = 'twitter_with_two_dataset_model_Random_Forest.pkl'
pickle.dump(classifier_rf, open(filename, 'wb'))

rf_score = accuracy_score(y_test, y_pred_rf)
print ('--' * 20)
print('Random Forest Accuracy: ', str(rf_score))
print('F1 score: ', f1_score(y_test, y_pred_rf, labels = [1,0]))
print ('--' * 20)
print ('')

# confusion_matrix
#Random Forest
y_pred_rf = classifier_rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_rf, labels = [1,0]), display_labels = [True, False])
cm_display.plot()
plt.show()

rf_score = accuracy_score(y_test, y_pred_rf)
print ('--' * 20)
print('Random Forest Accuracy: ', str(rf_score))
print('F1 score: ', f1_score(y_test, y_pred_rf, labels = [1,0]))
print ('--' * 20)
print ('')
from sklearn.metrics import plot_precision_recall_curve
plot_precision_recall_curve(classifier_rf, x_test, y_test, ax = plt.gca(),name = "Random Forest")

plt.title('Precision-Recall curve')
plt.plot(fpr_rf,tpr_rf,label="Random Forest, AUC="+str(auc_rf))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')

#add legend
plt.legend()

