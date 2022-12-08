import os, time, math
import numpy as np, nibabel as nib, pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from load_data import *

np.set_printoptions(threshold=np.inf,suppress=True)

n_slice = 21
image_size = 288
#model_path = 'model/network_model/CSN/model.ckpt-300'
model_path = 'model/network_model/CSN/best_model.ckpt'

testdataset_dir = 'data/raw_data/test_data'
predict_testdata_dir='data/process_data/csn_test_segmentation'
if not os.path.exists(predict_testdata_dir):
    os.makedirs(predict_testdata_dir)

testdata_list = select_test_data(testdataset_dir)
data_len = len(testdata_list['testing'])

if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))

        start_time = time.time()
        image_slice = np.zeros((image_size, image_size))

        for i in range(100):
            print(i)
            image_name = testdata_list['testing'][i]
            print(image_name)

            nim1 = nib.load(image_name)
            image_volume = nim1.get_data()

            image_shape=image_volume.shape

            raw_crop_image_volume=image_volume

            raw_shape=raw_crop_image_volume.shape
            print(raw_shape)

            crop_image_volume=raw_crop_image_volume

            crop_image_volume = normalization(crop_image_volume, (1.0, 99.0))
            crop_image_volume=crop_image_volume.astype(np.float32)

            # pad to 288 288 21
            X, Y, Z = crop_image_volume.shape
            cx, cy, cz = X / 2, Y / 2, Z / 2
            crop_image_volume_pad_3D,firstrow,firstcolumn,firstslice,lastrow,lastcolumn,lastslice = pad_image_for_csn(crop_image_volume, cx, cy, cz, image_size, n_slice)

            image_volume_pred = np.zeros((image_size, image_size,n_slice))

            image_volume_pred=crop_image_volume_pad_3D

            image_volume_pred = np.expand_dims(image_volume_pred, axis=0)
            pred_segt = sess.run('pred_segt_3D:0', feed_dict={'image_pl:0': image_volume_pred, 'training_pl:0': False})
            pred_segt = np.squeeze(pred_segt, axis=0).astype(np.int16)

            recover_crop_pred_volume=pred_segt
            recover_crop_pred_volume=recover_crop_pred_volume.astype(np.uint8)

            pad_pred_volume = np.zeros((image_shape))
            pad_pred_volume= recover_crop_pred_volume[firstrow:lastrow, firstcolumn:lastcolumn, firstslice:lastslice]
            pad_pred_volume = pad_pred_volume.astype(np.uint8)


            # for j in range(pad_pred_volume.shape[-1]):
            #     plt.subplot(121)
            #     plt.imshow(image_volume[:,:,j])
            #     plt.subplot(122)
            #     plt.imshow(pad_pred_volume[:,:,j])
            #     plt.show()

            print(image_shape,pad_pred_volume.shape)

            nim2 = nib.Nifti1Image(pad_pred_volume, nim1.affine, nim1.header.copy())

            cc = math.ceil((i + 1) / 2) + 100
            cb = (i + 1) / 2 + 100
            if cc > cb:

                nib.save(nim2, '{0}/patient{1}_ED.nii.gz'.format(predict_testdata_dir,cc))

            else:
                nib.save(nim2, '{0}/patient{1}_ES.nii.gz'.format(predict_testdata_dir,cc))


#--------------------------------------------------------------------------------------------------------
#---------------------------------dtype_transform-----------------------------------------------------
for i in range(50):

    segmentationdataset_dir1 = 'data/process_data/csn_test_segmentation/patient{0}_ED.nii.gz'.format(i + 101)
    segmentationdataset_dir2 = 'data/process_data/csn_test_segmentation/patient{0}_ES.nii.gz'.format(i + 101)

    print(i)
    print(segmentationdataset_dir1)
    print(segmentationdataset_dir2)

    nim1 = nib.load(segmentationdataset_dir1)
    nim2 = nib.load(segmentationdataset_dir2)

    image1 = nim1.get_data()
    image2 = nim2.get_data()

    image3 = image1.astype(np.int8)
    image4 = image2.astype(np.int8)

    nim5 = nib.Nifti1Image(image3, nim1.affine)
    nim6 = nib.Nifti1Image(image4, nim2.affine)

    nib.save(nim5, 'data/process_data/csn_test_segmentation/patient{0}_ED.nii.gz'.format(i + 101))
    nib.save(nim6, 'data/process_data/csn_test_segmentation/patient{0}_ES.nii.gz'.format(i + 101))