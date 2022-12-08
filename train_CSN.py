import os, time, random, math
import numpy as np, nibabel as nib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior
tf.disable_eager_execution()

from network_architecture import *
from load_data import *
from objective_loss import *
from experiments import exp_CSN as exp

checkpoint_model_dir = exp.checkpoint_model_dir
if not os.path.exists(exp.main_model_root):
    os.makedirs(exp.main_model_root)
if not os.path.exists(checkpoint_model_dir):
    os.makedirs(exp.checkpoint_model_dir)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#tf.set_random_seed(1)
#random.seed(1)

if __name__ == '__main__':

    dataset = read_data(exp.raw_training_data_dir, exp.process_data_dir, exp.image_size, exp.n_slice, is_csn=exp.is_csn)
    images_train = dataset['images_train']   #(176, 288, 288, 21)
    gts_train = dataset['gts_train']
    images_validation = dataset['images_validation']  #(20, 288, 288, 21)
    gts_validation = dataset['gts_validation']

    # Numpy Array
    images_train_3D, gts_train_3D, images_validation_3D, gts_validation_3D = adjust_data_3D(images_train,
                                                                                                 gts_train,
                                                                                                 images_validation,
                                                                                                 gts_validation, exp.image_size,exp.n_slice)
    print(images_train_3D.shape,gts_train_3D.shape)

    with tf.Graph().as_default():

        image_pl = tf.placeholder(tf.float32, shape=[None,None, exp.image_size,exp.n_slice], name='image_pl')
        segt_pl = tf.placeholder(tf.int32, shape=[None, None, exp.image_size,exp.n_slice], name='segt_pl')
        training_pl = tf.placeholder(tf.bool, shape=[], name='training_pl')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate_pl')

        logits_segt = CSN(image_pl, exp.segt_class, exp.n_slice,training=training_pl,n_filter=[16, 32, 64,128, 256])

        loss_segt, accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all, pred_segt = tf_loss_accuary_3D(logits_segt,segt_pl,exp.segt_class)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_pl).minimize(loss_segt)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=5)
        best_saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(init)
        start_time = time.time()

        init_epoch = 0
        continue_training = exp.continue_training
        print('is_continue?')
        print(continue_training)
        if continue_training:
            last_saved_model, init_epoch = find_the_last_model(checkpoint_model_dir)
            print(last_saved_model, init_epoch)
            saver.restore(sess, last_saved_model)

        if not os.path.exists(exp.log_dir1):
            os.makedirs(exp.log_dir1)
        csv_name = os.path.join(exp.log_dir1, '{0}_start{1}_log.csv'.format(exp.experiment_name, init_epoch))
        f_log = open(csv_name, 'w')
        f_log.write(
            'epoch,time,train_loss,val_acc,val_dice_all,val_dice_0,val_dice_1,val_dice_2,val_dice_3,val_segt_loss\n')

        best_epoch=0
        best_dice_sum = 2.0
        learning_rate=exp.learning_rate

        for epoch in range(init_epoch,  exp.train_epoch):
            print('epoch {} / {}'.format(epoch, exp.train_epoch))
            learning_rate=learning_rate*exp.lr_decay_rate

            val_dice1=0
            val_dice2=0
            val_dice3=0
            val_loss_segt_all=0

            data_len = images_train_3D.shape[0] #176
            n_per_epoch = int(data_len / exp.train_batch_size) #44
            random_arr = random.sample(range(data_len), data_len)


            for iteration in range(n_per_epoch):
                print('iteration {0} / {1} , epoch {2} / {3}'.format(iteration, n_per_epoch, epoch, exp.train_epoch))
                start_time_iter = time.time()

                image_in_batch_index=random_arr[iteration * exp.train_batch_size: (iteration + 1) * exp.train_batch_size]
                print(image_in_batch_index)

                images = np.zeros((exp.train_batch_size, exp.image_size, exp.image_size,exp.n_slice))
                segt_labels = np.zeros((exp.train_batch_size, exp.image_size, exp.image_size,exp.n_slice))

                for v in range(exp.train_batch_size):
                    images[v, :, :,:] = images_train_3D[image_in_batch_index[v] , :, :,:]
                    segt_labels[v, :, :,:] = gts_train_3D[image_in_batch_index[v] , :, :,:]

                images, segt_labels = data_augmentation_3D(images, segt_labels,  flip=True, gamma=True)
                _,  train_loss_segt, dice_12, dice_22, dice_32 = sess.run(
                    [train_op,  loss_segt,  dice_1, dice_2, dice_3],
                    {image_pl: images, segt_pl: segt_labels,training_pl: True,learning_rate_pl:learning_rate})

                print('  Iteration {}, epoch {} / {}, {:.3f}s'.format(iteration, epoch, exp.train_epoch,time.time() - start_time_iter))
                print('  segt loss:\t\t{:.6f}'.format(train_loss_segt))
                print('  train Dice 1:\t\t{:.6f}'.format(dice_12))
                print('  train Dice 2:\t\t{:.6f}'.format(dice_22))
                print('  train Dice 3:\t\t{:.6f}\n'.format(dice_32))
                print(learning_rate)
                print(best_epoch,best_dice_sum)


            data_len_val = images_validation_3D.shape[0] #20
            n_per_epoch_val = int(data_len_val / exp.validation_batch_size) #5
            random_arr_validation = range(data_len_val)


            for iteration_val in range(n_per_epoch_val):
                print('  iteration {0} / {1} , epoch {2} / {3}'.format(iteration_val, n_per_epoch_val, epoch, exp.train_epoch))

                val_image_in_batch_index=random_arr_validation[iteration_val * exp.validation_batch_size: (iteration_val + 1) * exp.validation_batch_size]

                valid_images = np.zeros((exp.validation_batch_size, exp.image_size, exp.image_size,exp.n_slice))
                valid_labels = np.zeros((exp.validation_batch_size, exp.image_size, exp.image_size,exp.n_slice))

                for v in range(exp.validation_batch_size):
                    valid_images[v, :, :,:] = images_validation_3D[val_image_in_batch_index[v], :, :,:]
                    valid_labels[v, :, :,:] = gts_validation_3D[val_image_in_batch_index[v], :, :,:]

                val_loss_segt, validation_acc, validation_dice_0, validation_dice_1, validation_dice_2, validation_dice_3, validation_dice_all = sess.run([loss_segt,  accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all],{image_pl: valid_images, segt_pl: valid_labels,training_pl:False,learning_rate_pl:learning_rate})
                
                val_dice1=val_dice1+validation_dice_1
                val_dice2=val_dice2+validation_dice_2
                val_dice3=val_dice3+validation_dice_3
                val_loss_segt_all=val_loss_segt_all+val_loss_segt

            print('  epoch {} / {}, {:.3f}s'.format(epoch, exp.train_epoch,time.time() - start_time_iter))
            print('  training loss:\t\t{:.6f}'.format(train_loss_segt))
            print('  validation accuracy:\t\t{:.2f}%'.format(validation_acc * 100))
            print('  validation Dice all:\t\t{:.6f}'.format(validation_dice_all))
            print('  validation Dice 0:\t\t{:.6f}\n'.format(validation_dice_0))

            print('  validation Dice 1:\t\t{:.6f}'.format(val_dice1/n_per_epoch_val))
            print('  validation Dice 2:\t\t{:.6f}'.format(val_dice2/n_per_epoch_val))
            print('  validation Dice 3:\t\t{:.6f}'.format(val_dice3/n_per_epoch_val))
            print('  val segt loss:\t\t{:.6f}\n'.format(val_loss_segt_all / n_per_epoch_val))

            f_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'.format(
                epoch, time.time() - start_time, train_loss_segt,
                validation_acc,validation_dice_all,  validation_dice_0, val_dice1/n_per_epoch_val, val_dice2/n_per_epoch_val, val_dice3/n_per_epoch_val,
                val_loss_segt_all/n_per_epoch_val))
            f_log.flush()

            dice_sum=val_dice1/n_per_epoch_val+val_dice2/n_per_epoch_val+val_dice3/n_per_epoch_val
            if dice_sum>best_dice_sum:
                best_epoch = epoch
                best_dice_sum = dice_sum
                best_checkpoint_file = os.path.join(checkpoint_model_dir, 'best_model.ckpt')
                best_saver.save(sess, best_checkpoint_file)

            if epoch % 20 == 0:
                checkpoint_file = os.path.join(checkpoint_model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)

        print(best_epoch)
        print('{:.3f}s\n'.format(time.time() - start_time))

        f_log.close()
        sess.close()
    dataset.close()

    
    