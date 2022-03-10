import functools
import itertools
import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import concatenate as  merge
from tensorflow.keras.layers import * 
import tensorflow as tf

from methods.transnet import padding_mask
from methods.transnet import Encoder
from methods.transnet import denseblock
from methods.transnet import transition



def train():

    # 1. 数据信息


    max_features = 21
    maxlen1 = 51
    maxlen2 = 33
    batch_size = 64
    nb_classes = 2
    init_form = 'RandomUniform'
    learning_rate = 0.001
    nb_dense_block = 1
    nb_layers = 3
    nb_filter = 32
    growth_rate = 32
    filter_size_block1 = 13
    filter_size_block2 = 7
    filter_size_ori = 1
    dense_number = 32
    dropout_rate = 0.2
    dropout_dense = 0.3
    weight_decay = 0.0001
    nb_batch_size = 64



    print('Loading data...')
    from methods.dataprocess_train import getMatrixLabel
    train_file_name = "/home/aita/zhiyuan/deepphos/transphos/dataset/PELM_Y_data.csv"


    x_train1, y_train1 = getMatrixLabel(train_file_name, ('Y'), window_size=maxlen1)
    # x_test1, y_test1 = getMatrixLabel(test_file_name, ('S'), window_size=maxlen1)

    x_train2, _ = getMatrixLabel(train_file_name, ('Y'), window_size=maxlen2)

    # x_test2, y_test2 = getMatrixLabel(test_file_name, ('S'), window_size=maxlen2)

    # 2. 构造模型，及训练模型

    inputs1 = Input(shape=(maxlen1,), dtype='int32')

    embeddings1 = Embedding(max_features, 16)(inputs1)

    print("\n"*2)


    mask_inputs1 = padding_mask(inputs1)






    out_seq1 = Encoder(n_layers=4,d_model=16,num_heads=4,middle_units=256,max_seq_len=maxlen1)(embeddings1, mask_inputs1, False)


    


    x1 = Conv1D(nb_filter, filter_size_ori,
                        kernel_initializer = init_form,
                        activation='relu',
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay))(out_seq1)
    for block_idx in range(nb_dense_block - 1):
        x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay)
        # add transition
        x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                    weight_decay=weight_decay)
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                            dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
    x1 = Activation('relu',name='seq1')(x1)


    inputs2 = Input(shape=(maxlen2,), dtype='int32')
    embeddings2 = Embedding(max_features, 16)(inputs2)
    mask_inputs2 = padding_mask(inputs2)
    out_seq2 = Encoder(n_layers=4,d_model=16,num_heads=4,middle_units=256,max_seq_len=maxlen2)(embeddings2, mask_inputs2, False)
    x2 = Conv1D(nb_filter, filter_size_ori,
                        kernel_initializer = init_form,
                        activation='relu',
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay))(out_seq2)


    for block_idx in range(nb_dense_block - 1):
        x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate,filter_size_block2,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay)
        x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                    weight_decay=weight_decay)


    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate,filter_size_block2,
                            dropout_rate=dropout_rate,
                            weight_decay=weight_decay)

                            

    x2 = Activation('relu',name='seq2')(x2)


    from tensorflow.keras.layers import concatenate

    x = concatenate([x1, x2], axis=-2, name='contact_multi_seq')
    x = Flatten()(x)

    x = Dense(dense_number,
            name ='Dense_1',
            activation='relu',kernel_initializer = init_form,
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)

    x = Dense(nb_classes,
            name = 'Dense_softmax',
            activation='softmax',kernel_initializer = init_form,
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay))(x)



    model = Model(inputs=[inputs1,inputs2], outputs=x,name="transphos-net")
    print(model.summary())

    opt = Adam(learning_rate=0.0002, decay=0.00001)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])

    print('Train...')

    history = model.fit([x_train1,x_train2], y_train1,
            batch_size=batch_size,
            epochs=40,
            validation_split=0.2)
    
    modelname = "/home/aita/zhiyuan/deepphos/transphos/model/PELM_Y"

    model.save_weights(modelname+'.h5',overwrite=True)
if __name__ =="__main__":
    train()