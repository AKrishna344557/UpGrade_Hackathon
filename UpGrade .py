#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r'C:\Users\DELL\Downloads\train_product_data.csv')


# In[4]:


# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())


# In[4]:



# Display basic information about the dataset
print(data.info())


# In[5]:


# Display summary statistics of the dataset
print(data.describe())


# In[6]:


print(data.isnull().sum())


# In[5]:


sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# In[6]:


# Fill missing values in the 'brand' column with most used brand name
most_common_brand = data['brand'].mode()[0]
data['brand'] = data['brand'].fillna(most_common_brand)


# In[8]:


# Drop rows where 'discription' has missing values
data = data.dropna(subset=['description'])


# In[9]:


print(data.isnull().sum())
print(f"The most common brand used for filling missing values is: {most_common_brand}")


# In[10]:


data['retail_price'].fillna(data['retail_price'].median(), inplace=True)
data['discounted_price'].fillna(data['discounted_price'].median(), inplace=True)


# In[11]:


# Drop rows where 'image' and 'product specifications' column has missing values
data = data.dropna(subset=['image'])
data = data.dropna(subset=['product_specifications'])


# In[12]:


print(data.isnull().sum())


# In[13]:


# Check the distribution of product categories
category_counts = data['product_category_tree'].value_counts()
print(category_counts)


# In[14]:


plt.figure(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Distribution of Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[15]:


# Clean the text data
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    return text

data['description'] = data['description'].apply(clean_text)


# In[16]:


# Encode the target labels
label_encoder = LabelEncoder()
data['product_category_tree'] = label_encoder.fit_transform(data['product_category_tree'])


# In[17]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data['description'], data['product_category_tree'], test_size=0.2, random_state=42)


# In[18]:


plt.figure(figsize=(12, 6))
sns.countplot(y=data['product_category_tree'], order=data['product_category_tree'].value_counts().index)
plt.title('Distribution of Product Categories')
plt.show()


# In[19]:


# Visualize the distribution of product ratings
plt.figure(figsize=(12, 6))
sns.histplot(data['product_rating'], bins=20)
plt.title('Distribution of Product Ratings')
plt.show()


# In[20]:


pip install wordcloud


# In[21]:


from wordcloud import WordCloud


# In[22]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['description']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Product Descriptions')
plt.show()


# In[23]:


# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)


# In[25]:


# Predict on the validation set
y_val_pred_lr = lr_model.predict(X_val_tfidf)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_val_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_val, y_val_pred_lr))


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Predict on the validation set
y_val_pred_rf = rf_model.predict(X_val_tfidf)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_val, y_val_pred_rf))


# In[32]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[27]:


pip install tensorflow


# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[31]:


# Define the Deep Learning model
dl_model = Sequential()
dl_model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(256, activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(len(label_encoder.classes_), activation='softmax'))


# In[32]:


# Compile the model
dl_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[33]:


# Train the model
dl_model.fit(X_train_tfidf.toarray(), y_train, epochs=10, batch_size=32, validation_data=(X_val_tfidf.toarray(), y_val))


# In[34]:


# Predict on the validation set
y_val_pred_dl = np.argmax(dl_model.predict(X_val_tfidf.toarray()), axis=1)


# In[35]:


# Evaluate the model
print("Deep Learning Model Accuracy:", accuracy_score(y_val, y_val_pred_dl))
print("Deep Learning Model Classification Report:\n", classification_report(y_val, y_val_pred_dl))


# In[36]:


#fine tuning 
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)


# In[37]:


test_data = pd.read_csv(r'C:\Users\DELL\Downloads\test_data.csv')


# In[39]:


test_results = pd.read_excel(r'C:\Users\DELL\Downloads\test_results.xlsx')


# In[40]:


print(test_data.head())
print(test_results.head())


# In[41]:


test_data['description'] = test_data['description'].fillna('')
test_data['description'] = test_data['description'].apply(clean_text)
X_test_tfidf = tfidf_vectorizer.transform(test_data['description'])


# In[42]:


# Predict using Logistic Regression
y_test_pred_lr = lr_model.predict(X_test_tfidf)


# In[43]:


# Predict using Random Forest
y_test_pred_rf = rf_model.predict(X_test_tfidf)


# In[44]:


# Predict using Deep Learning model
y_test_pred_dl = np.argmax(dl_model.predict(X_test_tfidf.toarray()), axis=1)


# In[48]:



# Preprocess the description column for test data 
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_test_tfidf = tfidf_vectorizer.fit_transform(test_data['description'])


# In[50]:


label_encoder = LabelEncoder()
label_encoder.fit(data['product_category_tree'])


# In[55]:


# Ensure all labels are strings
data['product_category_tree'] = data['product_category_tree'].astype(str)
test_results['product_category_tree'] = test_results['product_category_tree'].astype(str)


# In[57]:


# Preprocess the description column for test data (assuming TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_test_tfidf = tfidf_vectorizer.fit_transform(test_data['description'])


# In[59]:


# Combine all labels from train and test to re-fit LabelEncoder
combined_labels = pd.concat([data['product_category_tree'], test_results['product_category_tree']])
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)


# In[60]:


# Encode the true test labels
y_test_true_encoded = label_encoder.transform(test_results['product_category_tree'])


# In[61]:


# Predict with Logistic Regression model
y_test_pred_lr = lr_model.predict(X_test_tfidf)


# In[64]:


print("Logistic Regression Test Accuracy:", accuracy_score(y_test_true_encoded, y_test_pred_lr))
print("Logistic Regression Test Classification Report:\n", classification_report(y_test_true_encoded, y_test_pred_lr))


# In[65]:


# Predict with Random Forest model
y_test_pred_rf = rf_model.predict(X_test_tfidf)


# In[67]:


#Evaluate Random Forest model
print("Random Forest Test Accuracy:", accuracy_score(y_test_true_encoded, y_test_pred_rf))
print("Random Forest Test Classification Report:\n", classification_report(y_test_true_encoded, y_test_pred_rf))


# In[68]:


# Predict with Deep Learning model (assuming you convert sparse matrix to array for deep learning)
y_test_pred_dl = np.argmax(dl_model.predict(X_test_tfidf.toarray()), axis=1)


# In[69]:


# Evaluate Deep Learning model
print("Deep Learning Model Test Accuracy:", accuracy_score(y_test_true_encoded, y_test_pred_dl))
print("Deep Learning Model Test Classification Report:\n", classification_report(y_test_true_encoded, y_test_pred_dl))


# In[70]:


from sklearn.metrics import accuracy_score, classification_report

# Check some predictions and true values
print("Sample True Labels:", y_test_true_encoded[:10])
print("Sample Logistic Regression Predictions:", y_test_pred_lr[:10])

# Evaluate Logistic Regression model
print("Logistic Regression Test Accuracy:", accuracy_score(y_test_true_encoded, y_test_pred_lr))
print("Logistic Regression Test Classification Report:\n", classification_report(y_test_true_encoded, y_test_pred_lr))

# Evaluate Random Forest model
print("Random Forest Test Accuracy:", accuracy_score(y_test_true_encoded, y_test_pred_rf))
print("Random Forest Test Classification Report:\n", classification_report(y_test_true_encoded, y_test_pred_rf))


# In[ ]:




