# v1.0 horovod main 
from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
import cv2
import pdb



from model.model2 import *
from utils.colored import *
from utils.tools import *
from utils.params import *

def extract_5_patches(img,height,width):
    h,w,c=img.shape
    dy=int((h-height)/2)
    dx=int((w-width)/2)

    p1=img[:height,:width,:]
    p2=img[-height:,-width:,:]
    p3=img[:height,-width:,:]
    p4=img[-height:,:width,:]
    p5=img[dy:dy+height,dx:dx+width,:]
    p6=img[dy-1:dy+height-1,dx-1:dx+width-1,:]
    p7=img[dy+1:dy+height+1,dx+1:dx+width+1,:]

    p=np.array([p1,p2,p3,p4,p5])
    # p=np.array([p5])

    return p


def pre_process(img,h,w):
    img=img[...,::-1]
    img=img.astype(np.float32)/255
    img=cv2.resize(img,(256,256))
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_stddev =[0.229, 0.224, 0.225]
    for ii in range(3):
        img[:,:,ii]=(img[:,:,ii]-imagenet_mean[ii])/imagenet_stddev[ii]
      

    # extract 10 patches
    img=img[...,::-1] # back to BGR
    p1=extract_5_patches(img,h,w)
    p2=extract_5_patches(cv2.flip(img,1),h,w)

    p=np.concatenate([p1,p2],0)

  

    return p

def pre_process_without_scale(img,h,w):
    img=img[...,::-1]
    img=img.astype(np.float32)
    img=cv2.resize(img,(256,256))
    imagenet_mean=[122.6789143406786,116.66876761696767,104.0069879317889]
    for ii in range(3):
        img[:,:,ii]=(img[:,:,ii]-imagenet_mean[ii])
      
    # extract 10 patches
    img=img[...,::-1] # back to BGR
    p1=extract_5_patches(img,h,w)
    p2=extract_5_patches(cv2.flip(img,1),h,w)

    p=np.concatenate([p1,p2],0)

  

    return p


def main():
    """Create the model and start the training."""
    args = get_arguments()
    h=args.h
    w=args.w
    ss=args.domain.split('-')
    if ss[0]=='D':
        args.list1=args.list1.replace('amazon.txt','dslr.txt')
    elif ss[0]=='W':
        args.list1=args.list1.replace('amazon.txt','webcam.txt')

    if ss[1]=='A':
        args.list2=args.list2.replace('dslr.txt','amazon.txt')
    elif ss[1]=='W':
        args.list2=args.list2.replace('dslr.txt','webcam.txt')

    # pdb.set_trace()

    print(toRed(args.domain))

    # construct data generator
    coord = tf.train.Coordinator()
    image=tf.placeholder(tf.float32,shape=[None,args.h,args.w,3])
    model = DeepDICDModel(args, 1., image, None, image, None)

    # Gets moving_mean and moving_variance update operations from
    # tf.GraphKeys.UPDATE_OPS

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_local = tf.local_variables_initializer()
    init = tf.global_variables_initializer()

    sess.run([init_local, init])

    main_model_var=[v for v in tf.global_variables() if 'model' in v.name]

    if os.path.isfile(args.snapshot_dir+'.meta'):
        print(toGreen(args.snapshot_dir))
        loader = tf.train.Saver(var_list=main_model_var)
        load(loader, sess, args.snapshot_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(toGreen(ckpt.model_checkpoint_path))
            loader = tf.train.Saver(var_list=main_model_var)
            load_step = int(os.path.basename(
                ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

   
    f=open(args.list2,'r')
    acc=0
    count=0
    for line in f.readlines():
        count+=1
        line=line.strip()
        ss=line.split(' ')
        img_=cv2.imread(os.path.join(args.img_dir,ss[0]))
        img=pre_process_without_scale(img_,args.h,args.w)
        feed_dict={image:img}
        pred_=sess.run(model.slabel_pred,feed_dict=feed_dict)
        pred_=np.mean(pred_,0)
        pred=np.argmax(pred_)
        if int(ss[1])==int(pred):
            print(toCyan(ss[0]+' '+ss[1]+' '+str(pred)))
            acc+=1
        else:
            print(toRed(ss[0]+' '+ss[1]+' '+str(pred)))
    print(toGreen('accuarcy is: ' 
        + str(acc) + '/' + str(count) + '='+ str(acc/count)))

    print(toRed('done!'))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()
