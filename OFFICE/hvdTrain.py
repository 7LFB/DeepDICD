# v1.0 horovod main 
# corresponding model.py

from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np


from model.model import *
from data.myDataloader import *
from utils.colored import *
from utils.tools import *
from utils.params import *

import horovod.tensorflow as hvd


def main():
    """Create the model and start the training."""
    hvd.init()
    args = get_arguments()
    args.snapshot_dir=args.snapshot_dir.replace('DeepDICD/','DeepDICD/'+args.model_name+'-'+args.domain+'-')
    print(toMagenta(args.snapshot_dir))
    ss=args.domain.split('-')
    if ss[0]=='D':
        args.list1=args.list1.replace('amazon.txt','dslr.txt')
    elif ss[0]=='W':
        args.list1=args.list1.replace('amazon.txt','webcam.txt')

    if ss[1]=='A':
        args.list2=args.list2.replace('dslr.txt','amazon.txt')
    elif ss[1]=='W':
        args.list2=args.list2.replace('dslr.txt','webcam.txt')

    print(toMagenta(args.list1))
    print(toMagenta(args.list2))

    start_steps=args.start_steps
    
    h=args.h
    w=args.w

    # construct data generator
    file1 = open(args.list1) 
    num1 = len(file1.readlines()) 
    file2 = open(args.list2) 
    num2 = len(file2.readlines())
    file1.close()
    file2.close() 

    steps_per_epoch=int((num1/(args.batch_size)))
    num_steps=int(steps_per_epoch*args.num_epochs)
    val_num_steps=int(num2/args.batch_size)

    print(toCyan('src domain: {:d}, tar domain {:d}'.format(num1,num2)))
    print(toCyan('steps_per_epoch x num_epochs:{:d} x {:d}'.format(steps_per_epoch,args.num_epochs)))

    # Chong
    # split_batch_size=int(args.batch_size/hvd.size()) 
    myDataloader=Dataloader(args.img_dir,args.list1,args.list2,args.batch_size,args.h,args.w,args.num_threads)

    src_img=myDataloader.simg_batch
    src_label=myDataloader.slabel_batch
    tar_img=myDataloader.timg_batch
    tar_label=myDataloader.tlabel_batch

    coord = tf.train.Coordinator()

    # Using Poly learning rate policy 
    baseLR1 = tf.constant(args.lr1)
    baseLR2 = tf.constant(args.lr2)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    # lr1 = tf.scalar_mul(baseLR1, tf.pow((1 - step_ph / num_steps), args.power))
    # lr2 = tf.scalar_mul(baseLR2, tf.pow((1 - step_ph / num_steps), args.power))
    lr1=baseLR1/tf.pow(1+0.001*step_ph/steps_per_epoch,0.75)
    lr2=baseLR2/tf.pow(1+0.001*step_ph/steps_per_epoch,0.75)


    # lr1=baseLR1
    # lr2=baseLR2
    # decay_steps=steps_per_epoch*10
    # lr1=tf.train.exponential_decay(baseLR1,step_ph,decay_steps,0.1,staircase=True)
    # lr2=tf.train.exponential_decay(baseLR2,step_ph,decay_steps,0.1,staircase=True)
    keep_prob=tf.placeholder(dtype=tf.float32,shape=())
    # loss_balance =1- tf.scalar_mul(1., tf.pow((1 - step_ph / num_steps), args.power))
    loss_balance=tf.constant(1.)
    # boundaries = [np.float32(np.int32((8/10) * num_steps)), np.float32(np.int((9/10) * num_steps))]
    # values = [0., 0.1, 0.2]
    # loss_balance = tf.train.piecewise_constant(step_ph, boundaries, values)
   
    model = DeepCoralModel(args,keep_prob,src_img,src_label,tar_img,tar_label)
    model.build_losses(loss_balance) # loss_balance
    model.build_outputs()
    summary_=model.build_summary()
    loss=model.loss

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.no_update_mean_var == True:
        update_ops = None
    else:
        print(toMagenta('updating mean and var in batchnorm'))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    all_trainable_var=[v for v in tf.global_variables()]
    fine_tune_var = [v for v in all_trainable_var if 'fc8' not in v.name ]
    fine_tune_var_weights=[v for v in fine_tune_var if 'weights' in v.name]
    fine_tune_var_bias=[v for v in fine_tune_var if 'bias' in v.name]
    retrain_var = [v for v in all_trainable_var if 'fc8' in v.name]
    retrain_var_weights=[v for v in retrain_var if 'weights' in v.name]
    retrain_var_bias=[v for v in retrain_var if 'bias' in v.name]


    with tf.control_dependencies(update_ops):
        opt1_1 = tf.train.MomentumOptimizer(lr1*hvd.size(), args.momentum)
        opt1_1 = hvd.DistributedOptimizer(opt1_1)
        grads1_1 = tf.gradients(loss, fine_tune_var_weights)
        train_op_1_1 = opt1_1.apply_gradients(zip(grads1_1, fine_tune_var_weights))
        
        opt1_2 = tf.train.MomentumOptimizer(2*lr1*hvd.size(), args.momentum)
        opt1_2 = hvd.DistributedOptimizer(opt1_2)
        grads1_2 = tf.gradients(loss, fine_tune_var_bias)
        train_op_1_2 = opt1_2.apply_gradients(zip(grads1_2, fine_tune_var_bias))
        
        opt2_1 = tf.train.MomentumOptimizer(lr2*hvd.size(), args.momentum)
        opt2_1 = hvd.DistributedOptimizer(opt2_1)
        grads2_1 = tf.gradients(loss, retrain_var_weights)
        train_op_2_1 = opt2_1.apply_gradients(zip(grads2_1, retrain_var_weights))

        opt2_2 = tf.train.MomentumOptimizer(2*lr2*hvd.size(), args.momentum)
        opt2_2 = hvd.DistributedOptimizer(opt2_2)
        grads2_2 = tf.gradients(loss, retrain_var_bias)
        train_op_2_2 = opt2_2.apply_gradients(zip(grads2_2, retrain_var_bias))
        
        train_op = tf.group(train_op_1_1,train_op_1_2,train_op_2_1,train_op_2_2)
    # Set up tf session and initialize variables. 
    # 
    config = tf.ConfigProto()#Chong
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list=str(hvd.local_rank())
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    init_local=tf.local_variables_initializer()
    init = tf.global_variables_initializer()

   
    # construct summary
    summary_.append(tf.summary.scalar(
        'train/lr1', lr1))
    summary_.append(tf.summary.scalar(
        'train/lr2', lr2))
    summary_.append(tf.summary.scalar(
        'train/loss_balance', loss_balance))

    summary_merged=tf.summary.merge(summary_)
    if hvd.rank()==0:
        FinalSummary = tf.summary.FileWriter(args.snapshot_dir,sess.graph)

    # init
    sess.run([init_local,init])
    bcast=hvd.broadcast_global_variables(0)
    sess.run(bcast)
    
    # Saver for storing checkpoints of the model.
    var=tf.global_variables()
    skip_var=['fc8']
    saver = tf.train.Saver(var_list=var, max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path and args.resume:
        loader = tf.train.Saver(var_list=var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    elif not args.not_load_pretrained:
        print(toRed('Restore from pre-trained model...' + args.restore_from))
        model.load_initial_weights(sess, args.restore_from, skip_var) #Chong:0531

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    acc2_history=0
    for step in range(start_steps,num_steps):
        start_time = time.time()
        feed_dict = {step_ph: step, keep_prob: 0.5}
        summary, total_loss, _= sess.run([summary_merged,loss,train_op], feed_dict=feed_dict)
        if hvd.rank()==0:
            FinalSummary.add_summary(summary, step)
            duration = time.time() - start_time
            remain_time=duration*(num_steps-step)/3600
            print('\r',toCyan('{:s}:{:d}-{:d}-{:d} total loss = {:.3f},({:.3f} sec/step, ERT: {:.3f})'.format(args.model_name+'-'+args.domain,step%steps_per_epoch, step//steps_per_epoch,args.num_epochs,total_loss, duration,remain_time)),end='')
            
            if step % args.test_every == 0:
                acc1,acc2=0,0
                for jj in range(val_num_steps):
                    feed_dict = {keep_prob: 1}
                    src_acc,tar_acc= sess.run([model.src_acc,model.tar_acc], feed_dict=feed_dict)
                    acc1+=np.sum(src_acc)
                    acc2+=np.sum(tar_acc)

                acc1=acc1/(val_num_steps*args.batch_size)
                acc2=acc2/(val_num_steps*args.batch_size)
                # pdb.set_trace()
                test_summary = tf.Summary()
                test_summary.value.add(tag='test/source_accuracy',simple_value= acc1)
                test_summary.value.add(tag='test/target_accuracy',simple_value= acc2)
                FinalSummary.add_summary(test_summary, step)
                
                if acc2>acc2_history:
                    save(saver, sess, args.snapshot_dir, step)
                    acc2_history=acc2

    coord.request_stop()
    coord.join(threads)
    sess.close()
    
if __name__ == '__main__':
    main()
