import re
import string
import contractions
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer

train1 = pd.read_csv('/content/drive/MyDrive/train_1.csv')
test1 = pd.read_csv('/content/drive/MyDrive/test_1.csv')
df = pd.read_csv('/content/drive/MyDrive/dataset2.csv')

df = df[['index','oh_label','Text']]
df.rename(columns = {'index':'id','oh_label':'label','Text':'tweet'}, inplace = True)

train2, test2 = train_test_split(df, test_size=0.3, random_state=10, shuffle=True)
train2 = train2[['id', 'label', 'tweet']]
test2 = test2[['id', 'tweet']]

train = pd.concat([train1, train2], ignore_index = True)
test = pd.concat([test1,test2], ignore_index = True)

def clean_text(df, text_field):
    # Convert text to lowercase
    df[text_field] = df[text_field].str.lower()

    # Remove URLs
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"\w+:\/\/\S+", "", elem))

    # Remove square brackets content
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'\[.*?\]', '', elem))

    # Remove parentheses content
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'\(.*?\)', '', elem))

    # Remove hashtags
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'#', ' ', elem))

    # Remove mentions
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'@[^\s]+', '', elem))

    # Remove 'rt' at the beginning of the string
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'^rt', '', elem))

    # Fix contractions
    df[text_field] = df[text_field].apply(lambda elem: contractions.fix(elem))

    # Remove punctuations and digits
    df[text_field] = df[text_field].apply(lambda elem: re.sub('[%s]' % re.escape(string.punctuation + string.digits), ' ', elem))

    # Remove new line characters
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'\n', ' ', elem))
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'\\n', ' ', elem))

    # Remove quotation marks
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"[''\"“”‘’…]", '', elem))

    # Remove HTML attributes
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'<[^>]+>', '', elem))

    # Remove non-English languages
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'[^a-zA-Z\s]', '', elem))

    # Remove emojis and symbols
    df[text_field] = df[text_field].apply(lambda elem: re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE).sub(r'', elem))

    # Strip whitespace
    df[text_field] = df[text_field].str.strip()

    return df

test_clean = clean_text(test, "tweet")
train_clean = clean_text(train, "tweet")

train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]
train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(train_majority), random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])

y = train_upsampled['label'].values
y = y.reshape(-1, 1)

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(train_upsampled['tweet']).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

cv.fit(train_upsampled['tweet'])
with open('model_and_cv.pkl', 'wb') as file:
    pickle.dump((classifier_rf, cv), file)

