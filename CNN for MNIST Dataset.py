#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.keras.datasets import mnist


# In[19]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[20]:


x_train.shape


# In[21]:


single_image=x_train[0]


# In[22]:


single_image.shape


# In[23]:


single_image


# In[24]:


single_image.max()


# In[25]:


single_image.min()


# In[26]:


plt.imshow(single_image)


# In[27]:


y_train


# In[28]:


from tensorflow.keras.utils import to_categorical


# In[29]:


y_train.shape


# In[30]:


y_example=to_categorical(y_train,num_classes=10)


# In[31]:


y_example.shape


# In[32]:


y_cat_test=to_categorical(y_test,num_classes=10)


# In[33]:


y_cat_train=to_categorical(y_train,num_classes=10)


# In[34]:


# normalisation

x_train=x_train/255
x_test=x_test/255


# In[35]:


scaled_image=x_train[0]


# In[37]:


scaled_image.max()


# In[38]:


plt.imshow(scaled_image)


# In[41]:


# batch_size,height,width,color_channel
x_train=x_train.reshape(60000,28,28,1)


# In[42]:


x_test=x_test.reshape(10000,28,28,1)


# In[43]:


from tensorflow.keras.models import Sequential


# In[44]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[45]:


model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1, 1),padding='valid',input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten()) # 28*28
model.add(Dense(128,activation='relu'))


#OUTPUT LAYER-MULTI ClASS hence Output suhould be 'Softmax'
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[46]:


from tensorflow.keras.callbacks import EarlyStopping


# In[47]:


early_stop=EarlyStopping(monitor='val_loss',patience=1)


# In[48]:


model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])


# In[51]:


metrics=pd.DataFrame(model.history.history)


# In[52]:


metrics


# In[53]:


metrics[['loss','val_loss']].plot()


# In[55]:


metrics[['accuracy','val_accuracy']].plot()


# In[56]:


model.metrics_names


# In[57]:


model.evaluate(x_test,y_cat_test,verbose=0)


# In[80]:


from sklearn.metrics import confusion_matrix,classification_report


# In[82]:


predictions = np.argmax(model.predict(x_test),axis=1)


# In[83]:


predictions


# In[84]:


y_cat_test.shape


# In[85]:


y_test


# In[86]:


print(classification_report(y_true=y_test,y_pred=predictions))


# In[87]:


confusion_matrix(y_test,predictions)


# In[89]:


import seaborn as sns


# In[90]:


plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)


# In[91]:


# How to add new image and check it (example)
my_num=x_test[7]
plt.imshow(my_num)


# In[92]:


#num_image,width,height,color_channel

np.argmax(model.predict(my_num.reshape(1,28,28,1)),axis=1)


# In[ ]:




