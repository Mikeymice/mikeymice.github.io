---
layout: post
title: Random Multimodal Deep Learn
date: 2019-09-02T14:37:44.000Z
categories: Data Science Ensemble Random Deep Learn 
---
# Random Multimodal Deep Learning - Ensemble method for Deep Learning Models

## Introduction

I am a big fan for the ensemble method. Instead of using mainly data as parameter, the ensemble methods use other less-powerful machine learning models as parameter. Good examples for ensemble methods is XGBoost. The beauty of the XGBoost is that it feeds a series of machine learning model to train on the information and when the model makes a mistake, the XGBoost will add more weight on the sample so the next model will try to avoid making the same mistake. The intuition makes sense. One good analogy is doing practice exams. Imagine you are writing a practice exam multiple times. When you got a question wrong, you put a star next to the question. Therefore, the next time you do the exam, you will pay more attention to the question!

Another Ensemble technique is called Bagging which is to use voting for classification. Random Forest is one good example of this technique. The intuition is to use number of base learning models to vote for classification, and the class earns the most vote will get assigned. This is also the fundamental idea behind Random Multimodal Deep Learning (RMDL).

## Random Multimodal Deep Learning

The Random Multimodal Deep Learning model was introduced by Kamran Kowsari et al[link](https://arxiv.org/abs/1805.01890). The model used several several different deep learning models included Deep Neural Network (DNN), Convolutional Neural Network (CNN), and Recurrent Neural Networks(RNN).

The Advantage of using RMDL is that we do not need to worry about constructing the **best** structure of deep learning model for answering the question. It is time-consuming to identify how many layers or how many filters should we use for the deep learning model. Sometimes, even the transferred model may not be the most suitable choice.



## Demo
We will use the Kaggle's mnist dataset for our demo and use the RMDL compared to a transfer model.

To install the `RMDL`([Github](https://github.com/kk7nc/RMDL))package use the following command
```
pip install RMDL
```

For the image classification task, we will use the following function

```
from RMDL import RMDL_Image
Image_Classification(x_train, y_train, x_test, y_test, shape, batch_size=128,
                     sparse_categorical=True, random_deep=[3, 3, 3],
                     epochs=[500, 500, 500], plot=True,
                     min_hidden_layer_dnn=1, max_hidden_layer_dnn=8,
                     min_nodes_dnn=128, max_nodes_dnn=1024,
                     min_hidden_layer_rnn=1, max_hidden_layer_rnn=5,
                     min_nodes_rnn=32, max_nodes_rnn=128,
                     min_hidden_layer_cnn=3, max_hidden_layer_cnn=10,
                     min_nodes_cnn=128, max_nodes_cnn=512,
                     random_state=42, random_optimizor=True, dropout=0.05)

```
The detail documentation can be found here [link](https://github.com/kk7nc/RMDL/tree/master/RMDL#image-classification)

The parameter, `random_deep`, is a list of integer that indicate how many random generated machine learning models for DNN, RNN and CNN. We can also further control how many parameters of the model by passing how many possible layers and nodes for each model.

For our purpose, I used the following code.

```python
from keras.datasets import mnist
from RMDL import RMDL_Image
import numpy as np
```

    Using TensorFlow backend.



```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test_D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train_D / 255.0
X_test = X_test_D / 255.0
number_of_classes = np.unique(y_train).shape[0]
shape = (28, 28, 1)
batch_size = 128
sparse_categorical = 0

n_epochs = [10, 10, 10]  ## DNN--RNN-CNN
Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN
RMDL.Image_Classification(X_train, y_train, X_test, y_test,shape,
                             batch_size=batch_size,
                             sparse_categorical=True,
                             random_deep=Random_Deep,
                             epochs=n_epochs)
```


With the code above, the `RMDL.Image_Classification` function generated 3 for each of DNN, RNN, and CNN models and each went through 10 epochs.

Here is **part** of the output from generating the CNN mode

    CNN  0

    <keras.optimizers.Adam object at 0x7fd2662c2e10>
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/10
     - 46s - loss: 0.1425 - acc: 0.9539 - val_loss: 0.0378 - val_acc: 0.9872
     ... omitted
    Epoch 10/10
     - 38s - loss: 0.0094 - acc: 0.9969 - val_loss: 0.0251 - val_acc: 0.9937

    Epoch 00010: val_acc did not improve from 0.99440


    CNN  1
    ... ommited
    Epoch 10/10
     - 92s - loss: 14.5176 - acc: 0.0993 - val_loss: 14.4547 - val_acc: 0.1032


    CNN 2
    ... ommited
    Epoch 10/10
     - 46s - loss: 14.4711 - acc: 0.1022 - val_loss: 14.4902 - val_acc: 0.1010

    Epoch 00010: val_acc did not improve from 0.10100


    (10000, 9)
    (10000, 9)
    Accuracy of 9 models: [0.9844, 0.98, 0.1032, 0.9839, 0.9894, 0.9877, 0.9944, 0.1032, 0.101]
    Accuracy: 0.9924
    F1_Micro: 0.9924
    F1_Macro: 0.9923475636334604
    F1_weighted: 0.9923995418573717

With total **9** deep learning models, the RMDL can achieve **0.9924** validation accuracy! This is very impressive given that I did not provide any information of how to construct the deep learning models. I omit the pain of going though each layers and trying to tune the hyper parameters!

However, there are some limitations with **RMDL**

At the time of writing this blog, the RMDL package is only for demonstration only. The author did not provide any functionality to reuse the model for future prediction. The function, `Image_Classification`, only creates the model to train and get the accuracy, and it will delete the model and call the garbage collector! The author has an issue which promise that function for model resue. [Link](https://github.com/kk7nc/RMDL/issues/13)

Currently, it is possible to use `BuildModel.Build_Model_DNN_Image`([source](https://github.com/kk7nc/RMDL/blob/master/RMDL/BuildModel.py)) and other methods to get the model. However, this brings the second limitation. It will take alot of memory space to load all the random generated deep learning models for future prediction. Imagine you wish to generate 100 or 300 deep learning models. One possibility is to serialize the model and save to the hard drive. However, it will slow down the prediction process when we load the models one by one.

The Random Multimodal Deep Learn model is still a great approach with tons of potentials! There are many possible way to address the limitation such as tuning the hyper-parameters to reduce the number of parameters. Or only keeping the best 3 models with high accuracy for prediction/model reuse. I am also looking forward to the next version of the packagem with which the team will come out.

### Reference
 - https://github.com/kk7nc/RMDL
 - [Reference Paper](https://arxiv.org/abs/1805.01890)
