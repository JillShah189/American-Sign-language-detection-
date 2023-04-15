#!/usr/bin/env python
# coding: utf-8

# In[77]:


#import libraries 
import tensorflow as tf
import keras
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization,MaxPool2D
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# In[78]:


#import train data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets", labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(50, 50), shuffle=True, seed=123,
    validation_split=0.2, subset="training"
)


# In[79]:


#import validation data
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets", labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(50, 50), shuffle=True, seed=123,
    validation_split=0.2, subset="validation"
)


# In[80]:

#listing the classes
import os
labels = os.listdir("C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets")
labels.sort()
print(labels)


# In[81]:


#visulaize data
fig, ax = plt.subplots()
ax.bar("data",40500 ,color= 'b', label='Data')
ax.bar("train",32400 ,color= 'r', label='Train')
ax.bar("val",8100 ,color='g', label='Val')
leg = ax.legend();


# In[82]:


train_data


# In[85]:

#previewing dataset
def preview_dataset(dataset, nrow=4, ncol=8):
    plt.figure(figsize=(ncol*2,nrow*2))
    i = 0
    for (image, label) in train_data.take(nrow*ncol):
        image = image.numpy()[0].reshape((50,50,3))/255.0
        plt.subplot(nrow,ncol,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(labels[label[0].numpy()])
        i += 1
    plt.show()


# In[86]:


print("Training Dataset")
preview_dataset(train_data)


# In[87]:


print("Validation Dataset")
preview_dataset(val_data)


# In[74]:

#resnet architecture 
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add, ReLU
from tensorflow.keras.models import Model

# Define input shape
input_shape = (50, 50, 3)

# Define input tensor
inputs = Input(shape=input_shape)

# Initial convolution layer
x = Conv2D(64, kernel_size=(3,3), padding='same')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

# Residual blocks
for _ in range(3):
    shortcut = x
    
    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add skip connection
    x = Add()([shortcut, x])
    x = ReLU()(x)
    
    x = MaxPooling2D(pool_size=(2,2))(x)

# Flatten and dense layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(27, activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)
model.summary()


# In[30]:

#compiling the model 
model.compile(optimizer='Adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')


# In[31]:

#monitors validation loss
call = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True)


# In[32]:

#training the model 
fit= model.fit(train_data,validation_data=val_data,epochs=20,callbacks=[call])


# In[33]:


model.evaluate(val_data)


# In[34]:


#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = fit.history['accuracy']
val_acc = fit.history['val_accuracy']
loss = fit.history['loss']
val_loss = fit.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[35]:


# serialize weights to HDF5
model.save("./resnet_model.h5")


# In[70]:


labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 
               10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 
               19: 't', 20: 'u', 21: 'unknown', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z' }


# In[38]:

#test 1
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/b/b_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[39]:

#test 2
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/r/r_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[40]:

#test 3 
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/a/a_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[60]:

#test 4
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/v/v_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[47]:

#test 5
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/w/w_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[76]:

#test 6
image_path = "C:/AU/AU third year/6th sem/cv/archive/Sign Language for Alphabets/x/x_10.jpg"
new_img = image.load_img(image_path, target_size=(50, 50))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(labels[prediction[0]])
plt.imshow(new_img)


# In[ ]:




