# low-res-face-recognition
A Two-Branch DCNN model for Face Recognition in low-resolution images (24x24) to determine employee presence.
An implementation of the paper - "Low Resolution Face Recognition for Employee Detection" by Hrishikesh Kusneniwar and Arsalan Malik, 2021.

# Versions
1) Python     - 3.8
2) Tensorflow - 2.5.0
3) Keras      - 2.4.3 

# Data Description
1) FERET - https://www.nist.gov/itl/products-and-services/color-feret-database
2) Georgia Tech Face Database - http://www.anefian.com/research/face_reco.htm
3) KomNET Face Dataset - https://data.mendeley.com/datasets/hsv83m5zbb/2 
## Train/Val/Test split
### FERET   
    subjects     -  111  
    total images - 1165 (15 per subject)  
    train images - 1110 (10 per subject)
    val images   -  222 ( 2 per subject)
    test  images -  333 ( 3 per subject)  
### Georgia Tech Face Database    
    subjects     -   48  
    total images -  720 (15 per subject)  
    train images -  480 (10 per subject)
    val images   -   96 ( 2 per subject)  
    test  images -  144 ( 3 per subject)
### KomNET Face Dataset    
    subjects     -   50  
    total images -  750 (15 per subject)  
    train images -  500 (10 per subject)
    val images   -  100 ( 2 per subject)
    test  images -  150 ( 3 per subject)
    
### Test Set 1
    total images -  627 (209 subjects, 3 per subject from the above datasets)
    
### Test Set 1
    total images - 1227 = 627 (in database) + 600 (not in database)

From all the images in the above dataset, cropped face images were extracted using the Multi-task Cascaded Convolutional Networks (MTCNN) package in Keras.  
'Sample Data' consists of the first ten subjects of the created dataset.
    
# Architecture
We make use of a two branch architecture [1] that has 
two DCNNs to extract features from low resolution probe
images and high resolution gallery images and map them to
a 512-dimensional common space. The DCNN used by us is
FaceNet [2] that is pretrained on the VGGFace2 dataset. The
FaceNet model was obtained from the Github respository of
Hiroki Taniai [3]. 

![image](https://user-images.githubusercontent.com/68325029/132954873-141b50d5-3668-4ada-a1fb-fe84d541b291.png)
<p align="center"> Fig. 1 </p>

The top branch of the model (Fig. 1) is
called high resolution feature extraction convolutional neural
network (HRFECNN) that takes HR face images ( I<sup>hr</sup> ) as its
input. The bottom branch of the model is called low resolution
feature extraction convolutional neural network (LRFECNN)
that takes the corresponding LR face images ( I<sup>lr</sup> ) as its input.
The HR images as well as the LR images have to be in 160 x
160 dimensions before they can be fed into the two branches
of the model. Bicubic interpolation is used to resize the image
if it is not of the required dimensions. The 512-dimensional
feature vectors of I<sup>hr</sup> and I<sup>lr</sup> are then extracted from the
last layers of HRFECNN and LRFECNN respectively to the
common space.

<p align="center">![image](https://user-images.githubusercontent.com/68325029/132955019-8454d31b-5451-438c-8fa6-7fecdd7a159e.png)</p>
<p align="center"> Fig. 2 </p>

The parameters of HRFECNN (top branch) are not updated
during training, i.e. they are frozen. Therefore, y<sup>hr</sup>
is fixed but y<sup>lr</sup>
(LRFECNN) is trained to minimize the distance
between the HR and LR mapped images of the same person in
the common space. The mean-squared-error (MSE) objective
function is used for training the model,

<p align="center">![image](https://user-images.githubusercontent.com/68325029/132955019-8454d31b-5451-438c-8fa6-7fecdd7a159e.png)</p>
<p align="center"> Fig. 3 </p>


FaceNet model and weights can be obtained here: https://drive.google.com/drive/folders/1vCWyI_M3KcEuOF2yuksS24bzSmPrj6VW?usp=sharing

# Results
## Classification accuracy of 100% was obtained on the dataset of 209 subjects.   
Given below is the visualization of the 2D feature vectors obtained by performing PCA analysis on the FaceNet outputs of 'Sample Data' train images.  
  
![WhatsApp Image 2021-08-23 at 7 05 23 PM](https://user-images.githubusercontent.com/68325029/130456700-d44280e1-046e-47f8-9a4e-cb5cba832c54.jpeg)

# References
1. E. Zangeneh, M. Rahmati, and Y. Mohsenzadeh, “Low resolution face
recognition using a two-branch deep convolutional neural network
architecture,” Expert Systems with Applications, vol. 139, p. 112854,
2020.
2. F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2015, pp. 815–
823.
3. H. Taniai, “keras-facenet,” 2018. [Online]. Available: https://github.
com/nyoki-mtl/keras-facenet
3. Astawa, I Nyoman Gede Arya (2020), “KomNET: Face Image Dataset from Various Media”, Mendeley Data, V2, doi: 10.17632/hsv83m5zbb.2

## Dataset
https://drive.google.com/drive/u/1/folders/1sf46i0vQTJlQ-UqRYZPYDCzjZ4_5Rfcq
