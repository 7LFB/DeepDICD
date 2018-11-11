# tools
import os
import sys
import math
import tensorflow as tf


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
   # print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)


def adaptation_factor(x):
    def factor(x):
        den=1+tf.math.exp(-10.*x)
        alpha=tf.div(2.,den)-1.
        return alpha
    alpha=tf.cond(x>=1.,lambda:1.,lambda:factor(x))
    return alpha
