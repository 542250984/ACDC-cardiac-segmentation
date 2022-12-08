# ACDC-cardiac-segmentation

### 1. Download dataset and dependent packages
You can download the ACDC dataset from https://acdc.creatis.insa-lyon.fr/, the dataset is arranged as shown in *'data/raw_data'*

The dependent python(3.6) packages are as follows (may be incomplete):  
* h5py 2.9.0  
* numpy 1.16.2  
* nibabel 2.4.0  
* tensorflow-gpu 2.0.0

### 2. Training the Coarse Segmentation Network (CSN)
Training the CSN network. The network parameters can be modified in the file (*experiments/exp_CSN.py*)  
```  
python train_CSN.py  
```
### 3. Predicting testset segmentation results with the CSN
Use the trained CSN network to predict the coarse segmentation results of testset, and automatically save the results in *'data/process_data/csn_test_segmentation'*

The CSN can be any network structure, and its segmentation accuracy does not need to be very high. It only needs to meet the requirement that the size of the testset coarse segmentation result is consistent with the size of the the corresponding testset image. After we get the coarse segmentation results and save them locally, we don't need to adjust the CSN network anymore, just adjust the RCN network.
```  
python CSN_pred_test.py
```  
### 4. Training the Resolution-Consistent Network (RCN)
Training the RCN network. The network parameters can be modified in the file (*experiments/exp_RCN.py*)
```  
python train_RCN.py
```  
### 5. Predicting testset segmentation results with the RCN
Use the trained RCN network to predict the segmentation results of testset, and automatically save the results in *'data/process_data/predict_testdata'*
```  
python RCN_pred_test.py
```  

### Citation
If you find this helpful, please cite the paper. Thanks!

Yan Y, Chen C, Gao J. Lossless segmentation of cardiac medical images by a resolution consistent network with nondamage data preprocessing. Multimedia Tools and Applications (2022). https://doi.org/10.1007/s11042-022-14202-2
