# Leukemia-classification
Extracted features of test dataset are provided in .hdf5 format.
The studies were carried out with an Nvidia RTX 2080 Ti graphics processing unit having 11 GB memory, and the Keras framework.

#Description:-

(1) features_VGG16_SD:- This folder contains features extracted by VGG16 in .hdf5 format of subject-dependent (SD) dataset.

(2) features_test_VGG16_3CL_80_20_SI:- This folder contains features extracted by VGG16 with last three convolutional layers fine-tuned in .hdf5 format for 80-20 data split of subject-independent (SI) test dataset.

(3) features_test_VGG16_50_50_SI:- This folder contains features extracted by VGG16 in .hdf5 format for 50-50 data split of subject-independent (SI) test dataset.

(4) features_test_VGG16_80_20_SI:- This folder contains features extracted by VGG16 in .hdf5 format for 80-20 data split of subject-independent (SI) test dataset.

(5) RF_VGG16_3CL_80_20_SI:- This folder contains random forest (RF) models trained on features extracted by VGG16 with last three convolutional layers fine-tuned for 80-20 data split of subject-independent (SI) dataset.

(6) RF_VGG16_50_50_SD:- This folder contains random forest (RF) models trained on features extracted by VGG16 for 50-50 data split of subject-dependent (SD) dataset.

(7) RF_VGG16_50_50_SI:- This folder contains random forest (RF) models trained on features extracted by VGG16 for 50-50 data split of subject-independent (SI) dataset.

(8) RF_VGG16_80_20_SD:- This folder contains random forest (RF) models trained on features extracted by VGG16 for 80-20 data split of subject-dependent (SD) dataset.

(9) RF_VGG16_80_20_SI:- This folder contains random forest (RF) models trained on features extracted by VGG16 for 80-20 data split of subject-independent (SI) dataset.

(10) Codes:- This folder contains test codes for testing the trained models for different test datasets (SI and SD) and for different data splits (50-50 and 80-20).

#Steps to be followed:- 

(1) Download all the zip files (codes, .hdf5 files of test datset, and trained models).

(2) Unzip all the zip files.

(3) Run the test codes by providing proper path to HDF5 test dataset and path to trained models directory.

eg. :- RF_test_80_20_SI.py test code can be executed by "python RF_test_80_20_SD.py --db <path to HDF5 test dataset>/features_VGG16.hdf5 \--models <path to trained models directory>/RF_VGG16_80_20_SD" (mentioned in the last lines of test codes).
