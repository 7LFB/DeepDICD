
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pdb
import os
import glob
import random
import cv2
import pdb


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


def cv2_resize_tf(x,h,w):
    def resize256(x):
        return cv2.resize(x,(h,w))
    return tf.py_func(resize256,[x],tf.float32)

class Dataloader(object):

    def __init__(self, img_dir, list1, list2, batch_size, h,w, num_threads,seed=0):
        
        self.img_dir = img_dir
        self.list1=list1
        self.list2=list2
        self.target_h=h
        self.target_w=w

        self.simg_batch  = None
        self.slabel_batch = None
        self.timg_batch=None
        self.tlabel_batch=None


        # Chong
        tf.set_random_seed(1)

        queue1 = tf.train.string_input_producer([self.list1], shuffle=True)
        queue2 = tf.train.string_input_producer([self.list2], shuffle=True)
        # Chong: must use two seperate TextLineReader!!!
        line_reader1 = tf.TextLineReader()
        line_reader2 = tf.TextLineReader()

        _, line1 = line_reader1.read(queue1)
        _, line2 = line_reader2.read(queue2)
        #pdb.set_trace()

        split_line1 = tf.string_split([line1],' ').values
        split_line2 = tf.string_split([line2],' ').values
        
        p1=tf.string_join([self.img_dir,'/', split_line1[0]])
        slabel=tf.string_to_number(split_line1[1],tf.int64)
        p2=tf.string_join([self.img_dir,'/', split_line2[0]])
        tlabel=tf.string_to_number(split_line2[1],tf.int64)
  
        simg=self.read_image(p1)
        timg=self.read_image(p2)

        
        simg=self.augment_image(simg)
        timg=self.augment_image(timg)
        # else:
        #     simg=self.augment_image_when_test(simg)
        #     timg=self.augment_image_when_test(timg)

        simg.set_shape( [self.target_h, self.target_w, 3])
        # slabel.set_shape([None, 1])
        timg.set_shape( [self.target_h, self.target_w, 3])
        # tlabel.set_shape([None,1])
        simg=self.pre_process(simg)
        timg=self.pre_process(timg)

        simg=simg[...,::-1]
        timg=timg[...,::-1]


        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 2048
        capacity = min_after_dequeue + (4+num_threads)* batch_size
        self.simg_batch, self.slabel_batch, self.timg_batch, self.tlabel_batch = tf.train.shuffle_batch([simg,slabel,timg,tlabel],
                    batch_size, capacity, min_after_dequeue, num_threads)

       
    def augment_image(self,image):

        # randomly resize 256 or 257
        rnd_resize=tf.random_uniform([],0,1)
        image=tf.cond(rnd_resize<0.5, lambda:cv2_resize_tf(image,256,256), lambda:cv2_resize_tf(image,257,257))

        # randomly center crop
        rndx=tf.cond(rnd_resize<0.5,
            lambda:tf.cast(30*tf.random_uniform([],0,1),tf.int32),
            lambda:tf.cast(31*tf.random_uniform([],0,1),tf.int32))
        rndy=tf.cond(rnd_resize<0.5,
            lambda:tf.cast(30*tf.random_uniform([],0,1),tf.int32),
            lambda:tf.cast(31*tf.random_uniform([],0,1),tf.int32))
        image_aug=tf.image.crop_to_bounding_box(image,rndx,rndy,227,227)

        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        image_aug  = tf.cond(do_flip < 0.5, lambda: tf.image.flip_left_right(image_aug), lambda: image_aug)
        
        return image_aug


    def augment_image_when_test(self,image):

        # resize 256
        image=cv2_resize_tf(image,256,256)
        image_aug=tf.image.crop_to_bounding_box(image,14,14,227,227)

        return image_aug

    

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length-3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # image  = tf.image.convert_image_dtype(image,  tf.float32) # convert into [0,1] auto
        # image  = image*255
        image=tf.cast(image,tf.float32)
        # image  = tf.image.resize_images(image,  [256, 256], tf.image.ResizeMethod.AREA,align_corners=True)
        # image = cv2_resize_tf(image,256,256)
        return image

    def pre_process(self,float_image):
        # imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_stddev = tf.constant([0.229, 0.224, 0.225])
        imagenet_mean=[122.6789143406786,116.66876761696767,104.0069879317889]
        
        minus_mean1 = tf.fill([self.target_h, self.target_w, 1], -imagenet_mean[0])
        minus_mean2 = tf.fill([self.target_h, self.target_w, 1], -imagenet_mean[1])
        minus_mean3 = tf.fill([self.target_h, self.target_w, 1], -imagenet_mean[2])
        
        stddev1 = tf.fill([self.target_h, self.target_w, 1], imagenet_stddev[0])
        stddev2 = tf.fill([self.target_h, self.target_w, 1], imagenet_stddev[1])
        stddev3 = tf.fill([self.target_h, self.target_w, 1], imagenet_stddev[2])
        
        minus_mean = tf.concat([minus_mean1, minus_mean2, minus_mean3], axis=2)
        stddev = tf.concat([stddev1, stddev2, stddev3], axis=2)
        
        float_image = float_image + minus_mean
        # float_image = float_image / stddev

        return float_image
