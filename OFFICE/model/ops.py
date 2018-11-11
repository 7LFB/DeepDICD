import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def upsample_nn(x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs

def conv_(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
    conv1 = conv_(x,     num_out_layers, kernel_size, 1)
    conv2 = conv_(conv1, num_out_layers, kernel_size, 2)
    return conv2

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def resconv(x, num_layers, stride):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv_(x,         num_layers, 1, 1)
    conv2 = conv_(conv1,     num_layers, 3, stride)
    conv3 = conv_(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv_(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

def upconv_(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    conv = conv_(upsample, num_out_layers, kernel_size, 1)
    return conv

def deconv_(x, num_out_layers, kernel_size, scale):
    p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
    return conv[:,3:-1,3:-1,:]

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def Domain_MMD(sf,tf):
    mean=tf.reduce_mean(sf-tf,0,keep_dims=True)
    l2=tf.reduce_sum(tf.square(mean))
    return l2


def Class_MMD(sfeat_,tfeat_,slabel,tlabel,num_classes,alpha=0):
    
    bs=tf.shape(sfeat_)[0]
    sfeat=tf.reshape(sfeat_,[tf.cast(bs,tf.int32),-1])
    tfeat=tf.reshape(tfeat_,[tf.cast(bs,tf.int32),-1])
    ones=tf.ones_like(sfeat)
    current_source_count=tf.unsorted_segment_sum(ones,slabel,num_classes)
    current_target_count=tf.unsorted_segment_sum(ones,tlabel,num_classes)

    current_positive_source_count=tf.maximum(current_source_count,tf.ones_like(current_source_count))
    current_positive_target_count=tf.maximum(current_target_count,tf.ones_like(current_target_count))

    mask_=tf.minimum(current_positive_source_count,current_positive_target_count)
    mask=mask_>1

    current_source_mean=tf.divide(tf.unsorted_segment_sum(data=sfeat,segment_ids=slabel,num_segments=num_classes),current_positive_source_count)
    current_target_mean=tf.divide(tf.unsorted_segment_sum(data=tfeat,segment_ids=tlabel,num_segments=num_classes),current_positive_target_count)
    
    # same label 
    common_source_mean=tf.boolean_mask(current_source_mean,mask)
    common_target_mean=tf.boolean_mask(current_target_mean,mask)

    intra_class_loss=tf.reduce_mean(tf.square(common_source_mean-common_target_mean))

    # diff label
    # inter_class_loss=tf.reduce

    return intra_class_loss 


def centroid_mean(sfeat_,tfeat_,slabel,tlabel,num_classes,alpha=0):
    bs=tf.shape(sfeat_)[0]
    sfeat=tf.reshape(sfeat_,[tf.cast(bs,tf.int32),-1])
    tfeat=tf.reshape(tfeat_,[tf.cast(bs,tf.int32),-1])
    ones=tf.ones_like(sfeat)
    current_source_count=tf.unsorted_segment_sum(ones,slabel,num_classes)
    current_target_count=tf.unsorted_segment_sum(ones,tlabel,num_classes)

    current_positive_source_count=tf.maximum(current_source_count,tf.ones_like(current_source_count))
    current_positive_target_count=tf.maximum(current_target_count,tf.ones_like(current_target_count))

    current_source_mean=tf.divide(tf.unsorted_segment_sum(data=sfeat,segment_ids=slabel,num_segments=num_classes),current_positive_source_count)
    current_target_mean=tf.divide(tf.unsorted_segment_sum(data=tfeat,segment_ids=tlabel,num_segments=num_classes),current_positive_target_count)
    
    return current_source_mean, current_target_mean

# def intra_loss(source_mean,target_mean):
#     loss=tf.reduce_mean(tf.square(source_mean-target_mean))
#     return loss 

# def inter_loss(source_mean,target_mean,num_classes):
#     rnd=tf.random_uniform([],1,num_classes-1)
#     rnd=tf.cast(rnd,tf.int32)
#     source_mean_roll=tf.manip.roll(source_mean,shift=rnd,axis=0)
#     target_mean_roll=tf.manip.roll(target_mean,shift=rnd,axis=0)

#     source_loss=tf.reduce_mean(tf.square(source_mean-source_mean_roll))
#     target_loss=tf.reduce_mean(tf.square(target_mean-target_mean_roll))

#     return source_loss+target_loss


def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=[1,2,3])
    return output


def scale_gradient(x, scale, scope=None, reuse=None):
    with tf.name_scope('scale_grad'):
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
    return output


def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output


def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)


def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output


def perturb_image(x, p, classifier, pert='vat', scope=None):
    with tf.name_scope(scope, 'perturb_image'):
        eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

        # Predict on randomly perturbed image
        eps_p = classifier(x + eps, phase=True, reuse=True)
        loss = softmax_xent_two(labels=p, logits=eps_p)

        # Based on perturbed image, get direction of greatest error
        eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

        # Use that direction as adversarial perturbation
        eps_adv = normalize_perturbation(eps_adv)
        x_adv = tf.stop_gradient(x + args.radius * eps_adv)

    return x_adv


def vat_loss(x, p, classifier, scope=None):
    with tf.name_scope(scope, 'smoothing_loss'):
        x_adv = perturb_image(x, p, classifier)
        p_adv = classifier(x_adv, phase=True, reuse=True)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(p), logits=p_adv))

    return loss

def get_perturb_image(x,pred,classifier):

    eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

    # Predict on randomly perturbed image
    _, eps_p = classifier(x + eps, reuse=True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=eps_p,labels=tf.nn.softmax(pred)))

    # Based on perturbed image, get direction of greatest error
    eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

    # Use that direction as adversarial perturbation
    eps_adv = normalize_perturbation(eps_adv)
    x_adv = tf.stop_gradient(x + 3.5 * eps_adv)

    return x_adv


