# train with unet + NbNet
# modify from towerTrain3
# add label information
from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np


from model.model4_2 import *
from model.average_gradients import *
from data.threadBatchGenerator4_2 import *
from utils.colored import *
from utils.tools import *
from utils.params6 import *


def main():
    """Create the model and start the training."""
    args = get_arguments()
    args.snapshot_dir=args.snapshot_dir.replace('Demorph/','Demorph/'+args.model_name+'-')
    print(toMagenta(args.snapshot_dir))
    start_steps = args.start_steps

    fparams=open(os.path.join(args.snapshot_dir,'params.txt'),'w')
    fparams.write(args)
    fparams.close()
    h, w = map(int, args.input_size.split(','))
    args.h=h
    args.w=w

    # construct data generator
    trainFile = open(args.train_img_list)
    train_num_images = len(trainFile.readlines())
    valFile = open(args.val_img_list)
    val_num_images = len(valFile.readlines())
    trainFile.close()
    valFile.close()

    steps_per_epoch = int((train_num_images / args.batch_size))
    num_steps = int(steps_per_epoch * args.num_epochs)
    val_num_steps = int(val_num_images / args.batch_size)

    print(toCyan('train images: {:d}, test images {:d}'.format(
        train_num_images, val_num_images)))
    print(toCyan('steps_per_epoch x num_epochs:{:d} x {:d}'.format(
        steps_per_epoch, args.num_epochs)))

    myTrainBatchGenerator = BatchGenerator(
        args.train_img_list, args.batch_size, h, w, args.data_balance)()
    myValBatchGenerator = BatchGenerator(
        args.val_img_list, args.batch_size, h, w, 0.8)() # for same criterion
    

    coord = tf.train.Coordinator()

    # construct model
    img_morph = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 3])
    img_live = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 3])
    img_A = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 3])
    img_B = tf.placeholder(tf.float32,shape=[args.batch_size,h,w,3])
    labels=tf.placeholder(tf.float32,shape=[args.batch_size,1])
    # weights=tf.placeholder(tf.float32,shape=[1,h,w,1])
    # is_training = tf.placeholder(tf.bool,name='is_training')

    img_morph_splits = tf.split(img_morph, args.num_gpus, 0)
    img_live_splits = tf.split(img_live, args.num_gpus, 0)
    img_A_splits = tf.split(img_A, args.num_gpus, 0)
    img_B_splits = tf.split(img_B, args.num_gpus, 0)
    labels_splits=tf.split(labels,args.num_gpus,0)
   

    # Using Poly learning rate policy
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow(
        (1 - step_ph / num_steps), args.power))
    opt_step = tf.train.MomentumOptimizer(learning_rate, args.momentum)

    # construct model
    tower_grads = []
    tower_losses = []

    for i in range(args.num_gpus):
       with tf.device('/gpu:%d' % i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
               model = DemorphModel(args, img_morph_splits[i], img_live_splits[i], img_A_splits[
                                    i], img_B_splits[i], labels_splits[i])
               model.build_losses(args.loss_balance,args.dist_loss_balance)
            if i == 0:
                train_summary = model.build_summary('train')
                val_summary = model.build_summary('val')
            loss_ = model.loss
            all_trainable =[v for v in tf.trainable_variables() if 'losses' not in v.name]

            tower_losses.append(loss_)

            grads=opt_step.compute_gradients(loss_,all_trainable)

            tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    loss=tf.reduce_mean(tower_losses)


    # Gets moving_mean and moving_variance update operations from
    # tf.GraphKeys.UPDATE_OPS
    if args.no_update_mean_var == True:
        update_ops = None
    else:
        print(toMagenta('updating mean and var in batchnorm'))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = opt_step.apply_gradients(grads)

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print(toCyan('number of trainable parameters: {}'.format(total_num_parameters)))


    # Set up tf session and initialize variables.
    #
    config = tf.ConfigProto(allow_soft_placement=True)  # Chong
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    init_local = tf.local_variables_initializer()
    init = tf.global_variables_initializer()

    # construct summary
    train_summary.append(tf.summary.scalar(
        'train/learning_rate', learning_rate))
    train_summary.append(tf.summary.scalar('train/loss', loss))


    train_merged = tf.summary.merge(train_summary)
    val_merged = tf.summary.merge(val_summary)
    FinalSummary = tf.summary.FileWriter(args.snapshot_dir, sess.graph)

    # init
    sess.run([init_local, init])

    # Saver for storing checkpoints of the model.
    var = tf.global_variables()
    fine_tune_var=var
    saver = tf.train.Saver(var_list=var, max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path and args.resume:
        loader = tf.train.Saver(var_list=var)
        load_step = int(os.path.basename(
            ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    elif args.fine_tune_from is not None:
        print(toRed('Fine tune from pre-trained model...'))
        ckpt = tf.train.get_checkpoint_state(args.fine_tune_from)
        loader = tf.train.Saver(var_list=var)
        load_step = int(os.path.basename(
            ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    loss_history = 10000
    for step in range(start_steps, num_steps):
        start_time = time.time()
        img1, img2, img3, img4, labels_ = next(myTrainBatchGenerator)
        feed_dict = {img_morph: img1,
                     img_live: img2,
                     img_A: img3,
                     img_B: img4,
                     labels: labels_,
                     step_ph: step}

        summary, total_loss, _ = sess.run(
            [train_merged, loss, train_op], feed_dict=feed_dict)
        FinalSummary.add_summary(summary, step)
        duration = time.time() - start_time
        print('\r',toCyan('{:s}:{:d}-{:d}-{:d} total loss = {:.3f},({:.3f} sec/step)'.format(args.model_name,step %
                                                                                         steps_per_epoch, step // steps_per_epoch, args.num_epochs, total_loss, duration)),end='')

        if step % args.test_every == 0:
            losses = []
            for jj in range(val_num_steps):
                img1, img2, img3, img4, labels_= next(myValBatchGenerator)

                feed_dict = {img_morph: img1, 
                             img_live: img2,
                             img_A: img3,
                             img_B: img4,
                             labels: labels_}

                summary, total_loss = sess.run(
                    [val_merged, loss], feed_dict=feed_dict)
                losses.append(total_loss)
            FinalSummary.add_summary(summary, step)
            losses = np.array(losses)
            loss_ = np.mean(losses)

            test_summary = tf.Summary()
            test_summary.value.add(tag='val/loss', simple_value=loss_)
            FinalSummary.add_summary(test_summary, step)

            if loss_ < loss_history:
                save(saver, sess, args.snapshot_dir, step)
                loss_history = loss_

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
