
import tensorflow as tf
from tensorflow import keras

import numpy as np

from classCOCODetec import classCOCODetec
from utilsNeuralNetwork import *
        
def lossFunction(yTrue, yPred):
    
    sprob = 1 #Coeficiente probabilidad clase
    sconf = 1. #Coeficiente objeto
    snoob = 0.5 #Coeficiente no objeto
    scoor = 1 #Coeficiente coordenadas
    
    H, W = classCOCODetec.H, classCOCODetec.W
    C, B = classCOCODetec.C, classCOCODetec.B

    _probs = tf.reshape(yTrue[:,:,:,:C], [-1, H*W, C])
    _confs = tf.reshape(yTrue[:,:,:,C: C + B*C], [-1, H*W, C, B])
    _coord = tf.reshape(yTrue[:,:,:, C + B*C:], [-1, H*W, C, B, 4])

    _uno_obj = tf.reshape(tf.minimum(tf.reduce_sum(_probs, [2]), 1.0),[-1, H*W])

    
    net_out_probs = tf.reshape(yPred[:,:,:,:C], [-1, H*W, C])
    net_out_confs = tf.reshape(yPred[:,:,:,C: C + B*C], [-1, H*W, C, B])
    net_out_coords = tf.reshape(yPred[:,:,:,C + B*C:], [-1, H*W, C, B, 4])
                                                            
    adjusted_coords_xy = expit_tensor(net_out_coords[:,:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(tf.clip_by_value(net_out_coords[:,:,:,:,2:4],-15,8))* np.reshape(classCOCODetec.anchors, [1, 1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 1, 2]))
    adjusted_coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 4)

    adjusted_c = expit_tensor(net_out_confs)
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, C, B])
    
    adjusted_prob = expit_tensor(net_out_probs)
    adjusted_prob = tf.reshape(adjusted_prob,[-1, H*W, C])
    

    
    iou1 = tf.reshape(_coord, [-1, H, W, C*B, 4])
    iou2 = tf.reshape(adjusted_coords,[-1, H, W, C*B, 4])

    iou = calc_iou(iou1, iou2)
    ignoreMask = tf.reshape(tf.cast(iou<0.5, tf.float32), [-1, H*W, C, B])

    #bestBox = tf.cast(iou>=0.5, tf.float32)
    #confs = tf.reshape(tf.cast((iou >= bestBox), tf.float32),[-1, H*W, C, B])

    lossCoord = scoor*tf.reduce_mean(tf.reduce_sum(_confs*tf.reduce_sum(tf.square(adjusted_coords - _coord),[4]), [1,2,3]))

    #lossConfs = sconf*tf.reduce_mean(tf.reduce_sum(tf.squeeze(_confs) * tf.keras.losses.binary_crossentropy(_confs, adjusted_c),[1,2,3])) + \
     #           snoob*tf.reduce_mean(tf.reduce_sum((1 - tf.squeeze(_confs)) * ignoreMask * tf.keras.losses.binary_crossentropy(_confs, adjusted_c),[1,2,3]))
                
    lossConfs = sconf*tf.reduce_mean(tf.reduce_sum(_confs * tf.square(_confs - adjusted_c),[1,2,3])) + \
                snoob*tf.reduce_mean(tf.reduce_sum((1 - _confs) * ignoreMask * tf.square(_confs - adjusted_c),[1,2,3]))

    #lossClass = sprob*tf.reduce_mean(tf.reduce_sum(_uno_obj*tf.reduce_sum(tf.square(adjusted_prob - _probs),2),1))
    lossClass = sprob*tf.reduce_mean(tf.reduce_sum(_uno_obj*tf.keras.losses.binary_crossentropy(_probs, adjusted_prob),1))

    return lossCoord + lossConfs + lossClass

def neuralNetwork():

    x = tf.keras.Input(shape=(classCOCODetec.dim_fil,classCOCODetec.dim_col,3), name='input_layer')

    h_c1 = conv2d(inputs = x, f = 12, k = (3,3), s = 2, padding='same')
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(x)
    h_c1 = tf.keras.layers.concatenate([pool1, h_c1])

    h_c1 = conv2d(inputs = h_c1, f = 24, k = (3,3), s = 2, padding='same')
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(pool1)
    h_c1 = leaky_relu(tf.keras.layers.concatenate([pool2, h_c1]))

    h_c1 = batch_norm(conv2d(inputs = h_c1, f = 48, k = (3,3), s = 2))

    h_c1 = dense_layer(h_c1, 32, 64)
    h_c1 = leaky_relu(batch_norm(conv2d(inputs = h_c1, f = 256, k = (3,3), s = 1)))

    h_c1 = dense_layer(h_c1, 128, 256)

    h_c1 = conv2d(inputs = h_c1, f = classCOCODetec.C*(1 + classCOCODetec.B*(1+4)), k = (1,1), s = 1)

    model = tf.keras.Model(inputs=x, outputs=h_c1)

    return model, h_c1

if False:

    model, h_out = neuralNetwork()

    print('')
    print(model.summary())
    print('')
