# in build_classify_losses
#   tf.loss.sparse_softmax_cross_entropy vs tf.nn.softmax_cross_entropy_with_logits
# in build_batch_intra_losses
#   tf.nn.l2_loss vs tf.reduce(tf.square())
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

import termcolor
# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])

class DeepDICDModel(object):

    def __init__(self, params, keep_prob, simg, slabel,timg,tlabel=None):

        self.params = params
        self.simg=simg
        self.slabel=slabel
        self.timg=timg
        self.tlabel=tlabel
        self.keep_prob=keep_prob
        self.source_moving_centroid=tf.get_variable(name='source_moving_centroid',shape=[self.params.num_classes,256],initializer=tf.zeros_initializer(),trainable=False)
        self.target_moving_centroid=tf.get_variable(name='target_moving_centroid',shape=[self.params.num_classes,256],initializer=tf.zeros_initializer(),trainable=False)

        self.build_model()
         
    def build_model(self):
        with tf.variable_scope('model'):
                
            with tf.variable_scope('source') as encoder:
                self.sfc8,self.slabel_pred=self.encoder(self.simg)
                # self.soutput=outer(self.sfc7,self.ssoftmax)
            with tf.variable_scope(encoder,reuse=True):
                self.tfc8,self.tlabel_pred=self.encoder(self.timg)
                # self.toutput=outer(self.tfc7,self.tsoftmax)

    def build_outputs(self):
        self.src_acc=self.build_accuracy(self.slabel,self.slabel_pred)
        self.tar_acc=self.build_accuracy(self.tlabel,self.tlabel_pred)

    def encoder(self,x):

        fc8,score=self.build_alexnet(x)

        return fc8,score

    def build_losses(self,alpha1,alpha2):
        with tf.variable_scope('losses'):
         
            self.classify_loss_src=self.build_classify_losses(self.slabel,self.slabel_pred)
            self.classify_loss_tar=self.build_classify_losses(self.tlabel,self.tlabel_pred)
            self.build_adv_losses()

            src_update,tar_update=self.build_centroid_based_losses()

            self.build_regular_losses()

            # ------------- Loss Function ------------------
            # F_loss=model.classify_loss_src+model.dregular_loss+model.G_loss \
            # + model.class_wise_adaptation_loss \
            # + alpha2*(model.sintra_loss+model.tintra_loss-0.1*model.sinter_loss-0.1*model.tinter_loss)
            # F_loss=model.classify_loss_src+model.dregular_loss+alpha1*model.G_loss \
            # + alpha1*model.class_wise_adaptation_loss \
            # + alpha2*(model.sintra_loss+model.tintra_loss-0.1*model.sinter_loss-0.1*model.tinter_loss)
            self.F_loss=self.classify_loss_src \
                + self.gregular_loss \
                + alpha1 * self.G_loss \
                + alpha1 * self.class_wise_adaptation_loss \
                + alpha2 * (self.sintra_loss + self.tintra_loss) \
                # - 0.1*alpha2 * (self.sinter_loss + self.tinter_loss)
            return tf.group(src_update,tar_update)
    

    def build_accuracy(self,gt,pred):
        top_k_op=tf.nn.in_top_k(pred,gt,1)
        return top_k_op

    def build_summary(self):

        s1=tf.summary.scalar('train/source_loss',self.classify_loss_src)
        s2=tf.summary.scalar('train/target_loss',self.classify_loss_tar)

        i1=tf.summary.image('train/source_img',self.simg,1)
        i2=tf.summary.image('train/target_img',self.timg,1)

        l1=tf.summary.scalar('train/g_loss',self.G_loss)
        l2=tf.summary.scalar('train/dregular_loss',self.dregular_loss)
        l3=tf.summary.scalar('train/gregular_loss',self.gregular_loss)
        l4=tf.summary.scalar('train/class_wise_adaptation_loss',self.class_wise_adaptation_loss)
        
        c1=tf.summary.scalar('train/sintra_loss',self.sintra_loss)
        c2=tf.summary.scalar('train/tintra_loss',self.tintra_loss)
        c3=tf.summary.scalar('train/sinter_loss',self.sinter_loss)
        c4=tf.summary.scalar('train/tinter_loss',self.tinter_loss)

        h1=tf.summary.histogram('train/source_moving_centroid',self.source_moving_centroid)
        h2=tf.summary.histogram('train/target_moving_centroid',self.target_moving_centroid)
        
        return [s1,s2,i1,i2,l1,l2,l3,l4,c1,c2,c3,c4,h1,h2]

    def build_alexnet(self,x):
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
        flattened=flattened
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, self.keep_prob)
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, self.keep_prob)
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8=fc(fc7,4096,256,relu=False,name='fc8')
        score = fc(fc8, 256, self.params.num_classes, relu=False, stddev=0.005,name='fc9')
       
        return fc8, score
    
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

        return coral_loss

    def build_classify_losses(self,gt,pred):
        gt = tf.cast(gt, tf.int64)
        # Chong: vs 
        loss=tf.losses.sparse_softmax_cross_entropy(gt,pred)
        # gt = tf.one_hot(gt,self.params.num_classes) # Chong
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=gt))
        return loss

    def build_adv_losses(self):
        
        with tf.variable_scope('domain_descriminate') as DD:
            target_logits,_=Domain(self.tfc8)
        with tf.variable_scope(DD,reuse=True):
            source_logits,_=Domain(self.sfc8)
        
        D_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,labels=tf.ones_like(target_logits)))
        D_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logits,labels=tf.zeros_like(source_logits)))
        D_loss=D_real_loss+D_fake_loss
        G_loss=-D_loss
    
        self.G_loss=0.1*G_loss
        self.D_loss=0.1*D_loss
        

    def build_regular_losses(self):
        domain_var=[v for v in tf.trainable_variables() if 'domain' in v.name]
        generator_var=[v for v in tf.trainable_variables() if 'domain' not in v.name]
        
        print('--- domain var ---')
        for v in domain_var:
            print(v.name)

        print('--- generator var ---')
        for v in generator_var:
            print(v.name)

        self.dregular_loss=0.0005*tf.reduce_mean([tf.nn.l2_loss(v) for v in domain_var if 'weights' in v.name])
        self.gregular_loss=0.0005*tf.reduce_mean([tf.nn.l2_loss(v) for v in generator_var if 'weights' in v.name])

    def build_centroid_based_losses(self):

        source_label=self.slabel
        target_label=tf.argmax(self.tlabel_pred,1)
        ones=tf.ones_like(self.sfc8)
        current_source_count=tf.unsorted_segment_sum(ones,source_label,self.params.num_classes)
        current_target_count=tf.unsorted_segment_sum(ones,target_label,self.params.num_classes)

        current_positive_source_count=tf.maximum(current_source_count,tf.ones_like(current_source_count))
        current_positive_target_count=tf.maximum(current_target_count,tf.ones_like(current_target_count))

        current_source_centroid=tf.divide(tf.unsorted_segment_sum(data=self.sfc8,segment_ids=source_label,num_segments=self.params.num_classes),current_positive_source_count)
        current_target_centroid=tf.divide(tf.unsorted_segment_sum(data=self.tfc8,segment_ids=target_label,num_segments=self.params.num_classes),current_positive_target_count)

        decay=tf.constant(0.3)

        source_moving_centroid=(decay)*current_source_centroid+(1.-decay)*self.source_moving_centroid
        target_moving_centroid=(decay)*current_target_centroid+(1.-decay)*self.target_moving_centroid
        
        self.class_wise_adaptation_loss=self.build_class_wise_adaptation_losses(source_moving_centroid,target_moving_centroid)
        self.sintra_loss=self.build_batch_intra_losses(source_moving_centroid,self.sfc8,self.slabel)
        self.tintra_loss=self.build_batch_intra_losses(target_moving_centroid,self.tfc8,tf.argmax(self.tlabel_pred,1))
        self.sinter_loss=self.build_batch_inter_losses(source_moving_centroid,self.sfc8,self.slabel)
        self.tinter_loss=self.build_batch_inter_losses(target_moving_centroid,self.tfc8,tf.argmax(self.tlabel_pred,1))
           
        update_src = self.source_moving_centroid.assign(source_moving_centroid)
        update_tar = self.target_moving_centroid.assign(target_moving_centroid)
        
        return update_src, update_tar
   
    def build_class_wise_adaptation_losses(self,source_centroid,target_centroid):

        loss=tf.reduce_mean(tf.square(source_centroid-target_centroid))
        return loss
        
    def build_batch_intra_losses(self,centers,features,labels):
        len_features=features.get_shape()[1]
        # lables=tf.reshape(labels,[-1])
        centers_batch=tf.gather(centers,tf.cast(labels,tf.int32))
        
        # loss=tf.nn.l2_loss(features-centers_batch)
        loss=tf.reduce_mean(tf.square(features-centers_batch))
        return loss 
    
    def build_batch_inter_losses(self,centers,features,labels):

        ones=tf.ones_like(features)
        current_count=tf.unsorted_segment_sum(ones,labels,self.params.num_classes)
        current_positive_count=tf.maximum(current_count,tf.ones_like(current_count))
        current_centroid=tf.divide(tf.unsorted_segment_sum(data=features,segment_ids=labels,num_segments=self.params.num_classes),current_positive_count)
        mask=current_positive_count>1
        current_batch_centroid=tf.boolean_mask(current_centroid,mask)
        current_whole_centroid=tf.boolean_mask(centers,mask)

        nums=tf.cast(tf.shape(current_batch_centroid)[0],tf.float32)
        rnd=tf.random_uniform([],1.,nums-1)
        rnd=tf.cast(rnd,tf.int32)
        current_whole_centroid_roll=tf.manip.roll(current_whole_centroid,shift=rnd,axis=0)

        loss=tf.reduce_mean(tf.square(current_batch_centroid-current_whole_centroid_roll))
        return loss   

    def build_whole_inter_losses(self,centers,features,labels):
        pass

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
                            # print(toMagenta(var.name))
                            sess.run(var.assign(data))
    
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            # print(toMagenta(var.name))
                            sess.run(var.assign(data))
        






