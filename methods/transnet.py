import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import concatenate as  merge
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import * 
import tensorflow as tf
import numpy as np


def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, filter_size_block,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, axis=concat_axis)
        nb_filter += growth_rate
    return x


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, mode='sum', **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.mode = mode
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, x):
        if (self.embedding_dim == None) or (self.mode == 'sum'):
            self.embedding_dim = int(x.shape[-1])
        
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])
 
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2]) # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2]) # dim 2i+1
        
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        
        if self.mode == 'sum':
            return position_embedding + x
        
        elif self.mode == 'concat':
            position_embedding = tf.reshape(
              tf.tile(position_embedding, (int(x.shape[0]), 1)),
              (-1, self.sequence_len, self.embedding_dim)
            )
 
            return tf.concat([position_embedding, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.embedding_dim)

def padding_mask(seq):
    
    # 获取为 20的padding项
    seq = tf.cast(tf.math.equal(seq, 20), tf.float32)
 
    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)



def scaled_dot_product_attention(q, k, v, mask):
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
 
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
 
    return output
# 构造 multi head attention 层
 
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
 
        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        
        # 分头后的维度
        self.depth = d_model // num_heads
 
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
 
        self.dense = tf.keras.layers.Dense(d_model)
 
    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
 
    def call(self, inputs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]
 
        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)
 
        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
        # 通过缩放点积注意力层
        scaled_attention = scaled_dot_product_attention(q, k, v, mask) # (batch_size, num_heads, seq_len_q, depth)
        
        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)
 
        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
 
        # 全连接层
        output = self.dense(concat_attention)
        
        return output

def point_wise_feed_forward_network(d_model, middle_units):
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(d_model)])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, middle_units, epsilon=1e-6, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, middle_units)
        
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, mask, training):
        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs, mask])
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output) # (batch_size, input_seq_len, d_model)
        
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)
        
        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                max_seq_len, epsilon=1e-6, dropout_rate=0.1):
        super(Encoder, self).__init__()
 
        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)
 
        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads,
                    middle_units=middle_units,
                    epsilon=epsilon, dropout_rate=dropout_rate)
            for _ in range(n_layers)]
        
    def call(self, inputs, mask, training):
        emb = inputs
        emb = self.pos_embedding(emb)
        
        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb, mask, training)
 
        return emb


# 1. 数据信息

pass
max_features = 21
maxlen1 = 51
maxlen2 = 33
batch_size = 64
print('Loading data...')


def train():
    from dataprocess_train import getMatrixLabel
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

def transphos(x_train1,x_train2,y_train1,train=False):
    inputs1 = Input(shape=(maxlen1,), dtype='int32')

    embeddings1 = Embedding(max_features, 16)(inputs1)

    print("\n"*2)

    
    mask_inputs1 = padding_mask(inputs1)






    out_seq1 = Encoder(n_layers=4,d_model=16,num_heads=4,middle_units=256,max_seq_len=maxlen1)(embeddings1, mask_inputs1, False)

    print("\n"*2)
    print("out_seq:")
    print(out_seq1.shape)

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
    if train:
        history = model.fit([x_train1,x_train2], y_train1,
            batch_size=batch_size,
            epochs=40,
            validation_split=0.2)
    return model




