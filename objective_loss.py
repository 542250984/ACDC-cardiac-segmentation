import tensorflow.compat.v1 as tf

tf.disable_v2_behavior
import numpy as np

def dice_loss_3D(prob, label_1hot):
    tmp_0 = tf.reduce_sum(prob * label_1hot, [0, 1,2, 3])
    tmp_1 = tf.reduce_sum(prob + label_1hot, [0, 1,2, 3])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    loss = 1 - tf.reduce_mean(tmp_3)
    return loss

def tf_loss_accuary_3D(logits_segt, segt_pl,segt_class):

    prob_segt = tf.nn.softmax(logits_segt, dim=-1,name='prob_segt_3D')  # ([batch, Width, Height, n_slice, segt_class])
    pred_segt = tf.cast(tf.argmax(prob_segt, axis=-1), dtype=tf.int32,name='pred_segt_3D')  # ([batch, Width, Height,n_slice])

    # [batch, segt_class, Width, Height]
    label_1hot = tf.one_hot(indices=segt_pl, depth=segt_class,axis=-1)
    pred_1hot = tf.one_hot(indices=pred_segt, depth=segt_class,axis=-1)

    loss_segt = dice_loss_3D(prob_segt, label_1hot)

    accuracy_segt = tf_categorical_accuracy(pred_segt, segt_pl)
    dice_all = tf_dice_3D(pred_1hot, label_1hot)
    dice_0 = tf_dice_3D_single(pred_1hot[:, :,:, :,0], label_1hot[:, :,:, :,0])
    dice_1 = tf_dice_3D_single(pred_1hot[:, :,:, :,1], label_1hot[:, :,:, :,1])
    dice_2 = tf_dice_3D_single(pred_1hot[:, :,:, :,2], label_1hot[:, :,:, :,2])
    dice_3 = tf_dice_3D_single(pred_1hot[:, :,:, :,3], label_1hot[:, :,:, :,3])

    return loss_segt, accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all,pred_segt


def tf_loss_accuary_2D_channel_first(logits_segt, segt_pl,segt_class):

    prob_segt = tf.nn.softmax(logits_segt, dim=1,name='prob_segt_2D')  # ([batch, segt_class,Width, Height])
    pred_segt = tf.cast(tf.argmax(prob_segt, axis=1), dtype=tf.int32,name='pred_segt_2D')  # ([batch, Width, Height])

    # [batch, segt_class, Width, Height]
    label_1hot = tf.one_hot(indices=segt_pl, depth=segt_class,axis=1)
    pred_1hot = tf.one_hot(indices=pred_segt, depth=segt_class,axis=1)

    loss_segt = batch_dice_loss_2D_cf(prob_segt, label_1hot)

    accuracy_segt = tf_categorical_accuracy(pred_segt, segt_pl)
    dice_all = tf_dice_2D_cf(pred_1hot, label_1hot)
    dice_0 = tf_dice_2D_cf_single(pred_1hot[:, 0,:, :], label_1hot[:,0, :, :])
    dice_1 = tf_dice_2D_cf_single(pred_1hot[:, 1,:, :], label_1hot[:, 1,:, :])
    dice_2 = tf_dice_2D_cf_single(pred_1hot[:, 2,:, :], label_1hot[:, 2,:, :])
    dice_3 = tf_dice_2D_cf_single(pred_1hot[:, 3,:, :], label_1hot[:, 3,:, :])

    return loss_segt, accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all,pred_segt


def tf_dice_2D_cf(pred_1hot, label_1hot):
    tmp_0 = tf.reduce_sum(pred_1hot * label_1hot, [0,2, 3])
    tmp_1 = tf.reduce_sum(pred_1hot + label_1hot, [0,2, 3])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    loss = tf.reduce_mean(tmp_3)
    return loss

def tf_dice_2D_cf_single(pred_1hot, label_1hot):
    tmp_0 = tf.reduce_sum(pred_1hot * label_1hot, [0,1, 2])
    tmp_1 = tf.reduce_sum(pred_1hot + label_1hot, [0,1, 2])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    loss = tf.reduce_mean(tmp_3)
    return loss


def tf_dice_3D(pred_1hot, label_1hot):
    tmp_0 = tf.reduce_sum(pred_1hot * label_1hot, [0,1,2, 3])
    tmp_1 = tf.reduce_sum(pred_1hot + label_1hot, [0,1,2, 3])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    dice = tf.reduce_mean(tmp_3)
    return dice

def tf_dice_3D_single(pred_1hot, label_1hot):
    tmp_0 = tf.reduce_sum(pred_1hot * label_1hot, [0,1, 2,3])
    tmp_1 = tf.reduce_sum(pred_1hot + label_1hot, [0,1, 2,3])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    loss = tf.reduce_mean(tmp_3)
    return loss



def tf_categorical_accuracy(pred, truth):
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def batch_dice_loss_2D_cf(prob, label_1hot):
    tmp_0 = tf.reduce_sum(prob * label_1hot, [0, 2, 3])
    tmp_1 = tf.reduce_sum(prob + label_1hot, [0, 2, 3])
    tmp_3 = (2 * tmp_0) / (tmp_1 + 1e-8)
    loss = 1 - tf.reduce_mean(tmp_3)
    return loss

