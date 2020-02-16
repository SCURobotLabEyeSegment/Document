
# coding: utf-8

# In[1]:


from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# In[2]:


from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()


# In[3]:


x_train=x_train_image.reshape(60000,784).astype('float32')
x_test=x_test_image.reshape(10000,784).astype('float32')


# In[4]:


x_train_normalize=x_train/255
x_test_normalize=x_test/255


# In[5]:


y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)


# In[6]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[7]:


model = Sequential()


# In[8]:


model.add(Dense(units=1000,
               input_dim=784,
               kernel_initializer='normal',
               activation='relu'))


# In[9]:


model.add(Dropout(0.5))


# In[10]:


model.add(Dense(units=1000,
               input_dim=784,
               kernel_initializer='normal',
               activation='relu'))


# In[11]:


model.add(Dropout(0.5))


# In[12]:


model.add(Dense(units=10,
               kernel_initializer='normal',
               activation='softmax'))


# In[13]:


print(model.summary())


# In[14]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_history=model.fit(x=x_train_normalize,
                       y=y_train_onehot,validation_split=0.2,
                       epochs=10, batch_size=200,verbose=2)


# In[ ]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


# In[ ]:


show_train_history(train_history,'accuracy','val_accuracy')


# In[ ]:


show_train_history(train_history,'loss','val_loss')


# In[ ]:


scores=model.evaluate(x_test_normalize,y_test_onehot)
print()
print('accuracy=',scores[1])


# In[ ]:


prediction=model.predict_classes(x_test)


# In[ ]:


prediction


# In[ ]:


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


# In[ ]:


plot_images_labels_prediction(x_test_image,y_test_label,
                             prediction,idx=340)


# In[ ]:


import pandas as pd
pd.crosstab(y_test_label,prediction,
           rownames=['label'],colnames=['predict'])


# In[ ]:


df=pd.DataFrame({'label':y_test_label,'predict':prediction})
df[:2]


# In[ ]:


df[(df.label==5)&(df.predict==3)]


# In[ ]:


plot_images_labels_prediction(x_test_image,y_test_label,
                             prediction,idx=340,num=1)

