# DeepCoral V1.0
# 
# 
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
import pdb

from .zoo.alexnet import *
from .ops import *

class DeepCoralModel(object):

    def __init__(self, params, keep_prob, simg, slabel,timg,tlabel=None):

        self.params = params
        self.simg=simg
        self.slabel=slabel
        self.timg=timg
        self.tlabel=tlabel
        self.keep_prob=keep_prob

        self.source_centroid=tf.get_variable(name='source_centroid',shape=[self.params.num_classes,256],initializer=tf.zeros_initializer(),trainable=False)
        self.target_centroid=tf.get_variable(name='target_centroid',shape=[self.params.num_classes,256],initializer=tf.zeros_initializer(),trainable=False)

        self.build_model()
         
    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model'):
                
                with tf.variable_scope('source') as encoder:
                    self.slabel_pred=self.encoder(self.simg)
                with tf.variable_scope(encoder,reuse=True):
                    self.tlabel_pred=self.encoder(self.timg)
                
    def build_outputs(self):
        self.src_acc=self.build_accuracy(self.slabel,self.slabel_pred)
        self.tar_acc=self.build_accuracy(self.tlabel,self.tlabel_pred)

    def encoder(self,x):

        logits=self.build_alexnet(x)
        return logits
    
    def build_covariance_losses(self,feat1_,feat2_):
            bs=tf.shape(feat1_)[0]
            feat1=tf.reshape(feat1_,[tf.cast(bs,tf.int32),-1])
            feat2=tf.reshape(feat2_,[tf.cast(bs,tf.int32),-1])
            d=tf.cast(tf.shape(feat1)[1],tf.float32)

            xm=feat1-tf.reduce_mean(feat1,0,keep_dims=True)
            xc=tf.matmul(tf.transpose(xm),xm)/tf.cast(bs,tf.float32)

            xmt=feat2-tf.reduce_mean(feat2,0,keep_dims=True)
            xct=tf.matmul(tf.transpose(xmt),xmt)/tf.cast(bs,tf.float32)

            coral_loss=tf.reduce_sum(tf.multiply((xc-xct),(xc-xct)))
            coral_loss/=4*d*d

            class_loss=Class_MMD(feat1_,feat2_,self.slabel,tf.argmax(self.tlabel_pred,1),self.params.num_classes)


            return coral_loss + class_loss

    def build_classify_losses(self,gt,pred):
        gt = tf.cast(gt, tf.int64)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=gt, logits=pred))
        return loss

    def build_losses(self,alpha):
        with tf.variable_scope('losses'):
         
            self.classify_loss_src=self.build_classify_losses(self.slabel,self.slabel_pred)
            self.classify_loss_tar=self.build_classify_losses(self.tlabel,self.tlabel_pred)
            self.coral_loss=self.build_covariance_losses(self.slabel_pred,self.tlabel_pred)
            self.source_centroid_loss=self.get_center_losses('source_centroid',)
        
        self.loss= self.classify_loss_src+alpha*self.coral_loss

    def build_accuracy(self,gt,pred):
        top_k_op=tf.nn.in_top_k(pred,gt,1)
        return top_k_op

    def build_summary(self):

        s1=tf.summary.scalar('train/source_loss',self.classify_loss_src)
        s2=tf.summary.scalar('train/target_loss',self.classify_loss_tar)
        s3=tf.summary.scalar('train/coral_loss',self.coral_loss)

        i1=tf.summary.image('train/source_img',self.simg,1)
        i2=tf.summary.image('train/target_img',self.timg,1)
        # i3=tf.summary.image('train/trans_source_img',self.simg_,1)
        # i4=tf.summary.image('train/trans_target_img',self.timg_,1)
        return [s1,s2,s3,i1,i2]


    def build_alexnet(self,x):
        
        conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        # norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        # norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')
        # 3rd Layer: Conv (w ReLu)
        # conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, self.params.num_classes, relu=False, name='fc8')
        # fc8=tf.layers.dense(dropout7,self.params.num_classes,activation=tf.nn.sigmoid,name='fc8')
        return fc8

    def build_alexnet_v2(self,x):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 1, 1e-5, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name ='pool2')
        norm2 = lrn(pool2, 1, 1e-5, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        conv4_flattened=tf.contrib.layers.flatten(conv4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        self.flattened=flattened
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, self.keep_prob)
        self.fc6=fc6
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, self.keep_prob)
        self.fc7=fc7
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8=fc(fc7,4096,256,relu=False,name='fc8')
        self.fc8=fc8
        self.score = fc(fc8, 256, self.params.num_classes, relu=False, stddev=0.005,name='fc9')
        self.output=tf.nn.softmax(self.score)
        self.feature=self.fc8
        return self.score


    def load_initial_weights(self,sess,weights_path,SKIP_LAYER):
        """Load weights from file into network.
    
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(weights_path, encoding='bytes').item()
    
        # list of all assignment operators
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
    
            # Check if layer should be trained from scratch
            if op_name not in SKIP_LAYER:
    
                with tf.variable_scope('model/source/'+op_name, reuse=True):
    
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
    
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            sess.run(var.assign(data))
    
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            sess.run(var.assign(data))
    
    
    def get_center_losses(self,name,features,labels,alpha,num_classes):
        

        ones=tf.ones_like(features)
        len_features=features.get_shape()[1]
        lables=tf.reshape(labels,[-1])

        centers = tf.get_variable(name, [num_classes, len_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        
        centers_batch=tf.gather_nd(centers,labels)
        loss=tf.nn.l2_loss(featres-centers_batch)

        diff=centers_batch-features
        unique_label,unique_idx,unique_count=tf.unique_with_counts(labels)
        appear_times=tf.gather_nd(unique_count,unique_idx)
        appear_times=tf.reshape(appear_times,[-1,1])

        diff=diff/tf.cast(1+appear_times,tf.float32)
        diff=alpha*diff

        centers_update_op=tf.scatter_sub(centers,labels,diff)
        
        return loss, centers_update_op



  