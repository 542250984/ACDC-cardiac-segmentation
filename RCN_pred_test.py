import os, time, math
import numpy as np, nibabel as nib, pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from load_data import *

np.set_printoptions(threshold=np.inf,suppress=True)

n_slice = 21
image_size = 112
#model_path = 'model/network_model/RCN/model.ckpt-200'
model_path = 'model/network_model/RCN/best_model.ckpt'

testdataset_dir = 'data/raw_data/test_data'
csn_testdataset_dir='data/process_data/csn_test_segmentation'
predict_testdata_dir='data/process_data/predict_testdata'
if not os.path.exists(predict_testdata_dir):
    os.makedirs(predict_testdata_dir)
if not os.path.exists(csn_testdataset_dir):
    os.makedirs(csn_testdataset_dir)

testdata_list = select_test_data(testdataset_dir)
csn_test_list=select_csn_pred_test(csn_testdataset_dir)
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
            pred_name=csn_test_list['testing'][i]
            print(image_name)
            print(pred_name)

            nim1 = nib.load(image_name)
            image_volume = nim1.get_data()
            nim2=nib.load(pred_name)
            pred_volume=nim2.get_data()

            image_shape=image_volume.shape
            print(image_volume.shape,pred_volume.shape)


            raw_crop_image_volume,firstrow,lastrow,firstcolumn,lastcolumn,raw_crop_pred_volume=cardiac_center_point_positioning(image_volume,pred_volume,image_size)

            raw_shape=raw_crop_image_volume.shape
            print(raw_shape)

            crop_image_volume=raw_crop_image_volume
            crop_pred_volume=raw_crop_pred_volume

            crop_image_volume = normalization(crop_image_volume, (1.0, 99.0))
            crop_image_volume=crop_image_volume.astype(np.float32)

            image_volume_pred = np.zeros((image_size, image_size,crop_image_volume.shape[-1]))

            for slice in range(crop_image_volume.shape[-1]):
                image_slice=crop_image_volume[:,:,slice]
                images = np.expand_dims(image_slice, axis=0)
                pred_segt = sess.run('pred_segt_2D:0', feed_dict={'image_pl:0': images, 'training_pl:0': False})
                pred_segt = np.squeeze(pred_segt, axis=0).astype(np.int16)

                image_volume_pred[:,:,slice]=pred_segt

                # plt.subplot(121)
                # plt.imshow(image_slice)
                # plt.subplot(122)
                # plt.imshow(pred_segt)
                # plt.show()

            recover_crop_pred_volume=image_volume_pred
            recover_crop_pred_volume=recover_crop_pred_volume.astype(np.uint8)

            pad_pred_volume = np.zeros((image_shape))
            pad_pred_volume[firstrow:lastrow, firstcolumn:lastcolumn, :] = recover_crop_pred_volume
            pad_pred_volume = pad_pred_volume.astype(np.uint8)

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

    segmentationdataset_dir1 = 'data/process_data/predict_testdata/patient{0}_ED.nii.gz'.format(i + 101)
    segmentationdataset_dir2 = 'data/process_data/predict_testdata/patient{0}_ES.nii.gz'.format(i + 101)

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

    nib.save(nim5, 'data/process_data/predict_testdata/patient{0}_ED.nii.gz'.format(i + 101))
    nib.save(nim6, 'data/process_data/predict_testdata/patient{0}_ES.nii.gz'.format(i + 101))