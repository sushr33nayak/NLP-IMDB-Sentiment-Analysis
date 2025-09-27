#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Case Study-IMDB Sentiment Detection
# --------------------------------------
# 1. Load the the Given Data
# 2. Exploratory Data Analysis
# 3. Segregate the inputs and output
# 4. Split the sata into Training and Testing
# 5. Data Transformation-
#         a.Text Preprocessing(Special chart,stop words,lower case, lemmatization etc)
#         b.Text Vectorization-converting text to numerical vector-bag of words 
# 6. Model Building with training dataset
# 7. Prediction with test dataset
# 8. Evaluate the model performance


# In[2]:


import os
import numpy as np  # numerical/mathematical calculation
import pandas as pd # data manipulation
import matplotlib.pyplot as plt
import seaborn as sns # stats and visualisation
dataset=pd.read_csv("IMDB_dataset.csv")
dataset.head()


# In[4]:


from collections import Counter
words=" ".join(dataset['review'].values).split()
words_count=Counter(words)
common_words=words_count.most_common(20)
common_words_df=pd.DataFrame(common_words,columns=['Words','Count'])
sns.barplot(data=common_words_df,x='Count',y="Words")


# ## Step 3- Segregate inputs(x) and outpt(y)

# In[6]:


x=dataset[["review"]]
y=dataset[['sentiment']]
x


# y

# # Split data into train and test

# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)


# In[8]:


y_train.value_counts()


# In[9]:


y_test.value_counts()


# # Data Preparation on Training Data

# In[10]:


dataset['review'][0]


# In[11]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def clean(doc):
    regex="[^a-zA-Z]"
    doc=re.sub(regex," ",doc)
    doc=doc.lower()
    tokens=nltk.word_tokenize(doc)
    stop_words=list(set(stopwords.words('english')))
    cust_words=['watch','view','saw']
    stop_words=cust_words+stop_words
    filtered_tokens=[word for word in tokens if word not in stop_words]
    #lemmatizer
    lemmatizer=WordNetLemmatizer()
    lematized_token=[lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lematized_token)


# In[12]:


def tokenizer(doc):
    return nltk.word_tokenize(doc)



# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vector=CountVectorizer(preprocessor=clean,tokenizer=tokenizer,ngram_range=(1,1),binary=True)
get_ipython().run_line_magic('time', "x_train_transformed=bow_vector.fit_transform(x_train['review'])")

print()
print(x_train_transformed.shape)
print(f"Type of x_train:{type(x_train_transformed)}")
print(f"Vocal learn:{bow_vector.get_feature_names_out()[:20]}")


# In[14]:


new_train_ind=pd.DataFrame(x_train_transformed.toarray())
new_train_ind.head()


# In[15]:


from tqdm import tqdm,tqdm_notebook
tqdm.pandas()


# In[16]:


x_train['clean_review']=x_train['review'].progress_apply(lambda doc:clean(doc))
x_train.head()


# In[17]:


x_test_transformed = bow_vector.transform(x_test['review'])
x_test_transformed.shape


# In[19]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# In[20]:


classifier_nb = MultinomialNB()
classifier_nb.fit(x_train_transformed, y_train)


# In[21]:


x_train_transformed


# In[22]:


# prediction
y_pred_train = classifier_nb.predict(x_train_transformed)
y_pred_test = classifier_nb.predict(x_test_transformed)


# In[23]:


# evaluate the model performance
metrics.accuracy_score(y_train,y_pred_train)


# In[24]:


metrics.accuracy_score(y_test,y_pred_test)


# # RandomForest Model

# In[25]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train_transformed, y_train)


# In[26]:


# prediction
y_pred_train_rf = rf_model.predict(x_train_transformed)
y_pred_test_rf = rf_model.predict(x_test_transformed)


# In[63]:


metrics.accuracy_score(y_train,y_pred_train_rf)


# In[64]:


metrics.accuracy_score(y_test,y_pred_test)


# In[ ]:




