#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd                         # Pandas is a data manipulation and analysis library.
import tensorflow as tf                     # TensorFlow is an open-source machine learning library.
import os                                  # Operating system-related functionality.
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Keras ImageDataGenerator for data augmentation.
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing sets.
import matplotlib.pyplot as plt            # Plotting library.
from mpl_toolkits.axes_grid1 import ImageGrid  # Toolkit for creating image grids.
from pathlib import Path                   # Path manipulation library.
from PIL import Image                      # Python Imaging Library for image processing.
import cv2                                 # OpenCV library for computer vision tasks.
from tensorflow.keras import layers        # Layers module from Keras for building neural networks.
from sklearn import preprocessing          # Scikit-learn library for preprocessing tasks.
import splitfolders                        # Library for splitting data into train, validation, and test sets.
from tensorflow.keras.utils import to_categorical  # Utility function for one-hot encoding.
import numpy as np                          # Numerical computing library.
from tensorflow.keras.preprocessing import image  # Keras image preprocessing module.
import matplotlib.pyplot as plt            # Plotting library.


# In[2]:


image_link = list(Path(r'D:\messina\New folder\noise_clean_test_dataset\noisy_test_dataset').glob(r'**/*.jpg'))
image_name = [x.parents[0].stem for x in image_link]
image_label = preprocessing.LabelEncoder().fit_transform(image_name)


# In[3]:


df = pd.DataFrame()
df['link'] = np.array(image_link, dtype=str)
df['name'] = image_name
df['label'] = image_label


# In[11]:


df


# In[12]:


fig = plt.figure(1, figsize = (15,15))
grid = ImageGrid(fig, 121, nrows_ncols = (5,4), axes_pad = 0.10)
i = 0
for category_id, category in enumerate(df.name.unique()):
    for filepath in df[df['name'] == category]['link'].values[:4]:
        ax = grid[i]
        img = Image.open(filepath)
        ax.imshow(img)
        ax.axis('off')
        if i % 4 == 4-1:
            ax.text(300,100, category, verticalalignment = 'center', fontsize = 20, color ='red')
        i+=1
        
plt.show()


# In[4]:


# Split the data into training and testing sets
#train_df, test_df = train_test_split(df, test_size=0.3, random_state=1)


# In[5]:


# ImageDataGenerator for training without data augmentation
#train_datagen = ImageDataGenerator(rescale=1./255)
# ImageDataGenerator for testing without data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


# # Create data generators
# train_images = train_datagen.flow_from_dataframe(
# dataframe=train_df,
# x_col='link',
# y_col='name',
# target_size=(250, 250),
# batch_size=32,
# class_mode='categorical',
# subset='training'
# )
test_images = test_datagen.flow_from_dataframe(
dataframe=df,
x_col='link',
y_col='name',
target_size=(250, 250),
batch_size=32,
class_mode='categorical',
shuffle=False
)


# In[7]:


# # Define and train the model
# model = tf.keras.models.Sequential([
# tf.keras.layers.Conv2D(16, (3, 3), input_shape=(250, 250, 3), activation='relu'),
# tf.keras.layers.MaxPooling2D(2, 2),
# tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
# tf.keras.layers.MaxPooling2D(2, 2),
# tf.keras.layers.Flatten(),
# tf.keras.layers.Dense(64, activation='relu'),
# tf.keras.layers.Dense(128, activation='relu'),
# tf.keras.layers.Dense(5, activation='softmax')
# ])
# model.compile(
# optimizer='adam',
# loss='categorical_crossentropy',
# metrics=['accuracy']
# )

# # Train the model on the training data generator for a few epochs
# model.fit(train_images, epochs=10)


from keras.models import load_model
# Load the pre-trained model
loaded_model = load_model(r'D:\messina\New folder\trained120k_model.h5')


# Evaluate the model on the local test set
y_true = test_images.classes
y_pred_probabilities = loaded_model.predict(test_images)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_probabilities, axis=1)


# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import os
# Print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[9]:


# Print classification report
class_report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(class_report)


# In[10]:


# Calculate and print overall model performance score
performance_score = loaded_model.evaluate(test_images)
print("\nPerformance Score:")
print(f"Loss: {performance_score[0]}, Accuracy: {performance_score[1]}")

# If needed, you can further analyze metrics like precision, recall, and F1 score for individual classes using the values from the confusion matrix.


# In[ ]:




