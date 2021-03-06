
# coding: utf-8

# In[3]:


from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# In[7]:


(x_Train, y_Train),(x_Test, y_Test) =mnist.load_data()


# In[8]:


x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


# In[9]:


x_Train4D_normalize=x_Train4D/255
x_Test4D_normalize=x_Test4D/255


# In[10]:


y_TrainOneHot=np_utils.to_categorical(y_Train)
y_TestOneHot=np_utils.to_categorical(y_Test)


# In[13]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# In[14]:


model=Sequential()


# In[15]:


model.add(Conv2D(filters=16,
                kernel_size=(5,5),
                padding='same',
                input_shape=(28,28,1),
                activation='relu'))


# In[16]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[17]:


model.add(Conv2D(filters=36,
                kernel_size=(5,5),
                padding='same',
                activation='relu'))


# In[18]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[19]:


model.add(Dropout(0.25))


# In[20]:


model.add(Flatten())


# In[21]:


model.add(Dense(128,activation='relu'))


# In[22]:


model.add(Dense(10,activation='softmax'))


# In[23]:


print(model.summary())


# In[24]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])


# In[25]:


train_history=model.fit(x=x_Train4D_normalize,
                       y=y_TrainOneHot,validation_split=0.2,
                       epochs=10,batch_size=300,verbose=2)


# In[41]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


# In[42]:


show_train_history(train_history,'accuracy','val_accuracy')


# In[49]:


show_train_history(train_history,'loss','val_loss')


# In[50]:


scores=model.evaluate(x_Test4D_normalize, y_TestOneHot)
scores[1]


# In[53]:


prediction=model.predict_classes(x_Test4D_normalize)


# In[54]:


prediction[:10]


# In[56]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[57]:


plot_images_labels_prediction(x_Test, y_Test, prediction, idx=0)


# In[58]:


import pandas as pd
pd.crosstab(y_Test, prediction,
           rownames=['label'],colnames=['predict'])


# In[60]:


from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()


# In[61]:


plot_images_labels_prediction(x_test_image,y_test_label,
                             prediction,idx=340)

