# low-res-face-recognition
A Two-Branch DCNN model for Face Recognition in low-resolution images (24x24) to determine employee presence.
An implementation of the paper - "Low Resolution Face Recognition for Employee Detection" by Hrishikesh Kusneniwar and Arsalan Malik, 2021.

# Versions
1) Python     - 3.8
2) Tensorflow - 2.5.0
3) Keras      - 2.4.3

# Platform
The training was performed on a Laptop enabled with the
Nvidia GTX 1060 MaxQ GPU, and Intel Core i7-8750H CPU.
The time taken for training on the dataset of 209 subjects for
45 epochs is described below.

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
    
### Preprocessing 
In the preprocessing stage, the faces were
cropped using the python library, Multi-Task Cascaded Convolutional Neural (MTCNN). As shown in Figure 4, the image
to be processed was given a bounding box which was then
cropped based on the bounding box. After obtaining the face,
the cropped image was resized to 160 x 160 dimensions via
bicubic interpolation to be sent as input to our model. The
above steps were performed for all the images in the acquired
databases, constituting the high resolution gallery images.

<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132974532-e3263715-cfbe-4832-92f7-1cf9dcb37cbd.png">
</p>

### Training set
The training set consists of 2090 pairs of
images. Each pair contains a HR and LR version of a particular
image. After preprocessing, we took 10 face images of each
subject from our combined database, to form the HR images
in the 2090 pairs. Their LR counterparts were created by
downsampling the HR image using bicubic interpolation to
the desired dimensions.
### Validation set
The validation set consists of 418 pairs
of HR and LR images. After preprocessing, we took 2 face
images of each subject from our combined database, to form
the HR images in the 418 pairs. Their LR counterparts were
obtained via bicubic interpolation similar to the training set.
### Test set 1
The test set 1 contains 627 LR images that
were obtained by taking 3 face images of each subject from
our combined database. Bicubic interpolation was used for
obtaining the desired low resolution dimensions.
### Test set 2
The test set 2 contains the same 627 LR images
in test 1, as well as 600 additional LR face images of people
outside of our combined database of 209 subjects. These
600 ’out-database’ face images are obtained from Totally
Looks Like Data [5]. The same preprocessing steps and
downsampling as described earlier were performed on these
out-database images. Therefore, total size of test set 2 is 1227
LR images.
  
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

<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976615-ffb2233c-4b48-41b2-9928-e17c5fad41c3.png">
</p>
<p align="center"> Fig. 2 </p>

The parameters of HRFECNN (top branch) are not updated
during training, i.e. they are frozen. Therefore, y<sup>hr</sup>
is fixed but y<sup>lr</sup>
(LRFECNN) is trained to minimize the distance
between the HR and LR mapped images of the same person in
the common space. The mean-squared-error (MSE) objective
function is used for training the model,

<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976649-5887f78b-b71d-40d3-b115-1901c930b023.png">
</p>
<p align="center"> Fig. 3 </p>


FaceNet model and weights can be obtained here: https://drive.google.com/drive/folders/1vCWyI_M3KcEuOF2yuksS24bzSmPrj6VW?usp=sharing

# Training
Pairs of HR and LR images from the training set were used
to train the model. Weights of LRFECNN were updated via
gradient descent using the Adam Optimizer with a batch size
of 64 images. The weights of HRFECNN were kept fixed. We
performed the training for 45 epochs till there was insignificant
decrement in the training loss. The learning rate was decayed
as described in Table 1

<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976466-01434d8f-acc7-4e5f-8abb-ec15d6c6a406.png">
</p>
<p align="center"> Table 1</p>

After completion of training, the feature vectors of LR
images in the training set were obtained from the last layer
of LRFECNN. These feature vectors, in conjunction with
the true labels of the images were then used to train a
Logistic Regression classifier. We used the Logistic Regression
model available in the Scikit-Learn library, with the ’lbfgs’
solver. We trained the classifier for 10<sup>20</sup> iterations, which took
approximately 2 seconds to complete.

# Testing
The 512-dimensional feature vectors of LR probe images
were obtained from the last layer of LRFECNN. These feature
vectors were then fed into the Logistic Regression classifier to
obtain the predicted labels corresponding to the probe images.

# Results
## PCA visualization before training the LRFECNN
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976914-fb469f68-549d-4712-8611-219e5b40b433.png">
</p>
<p align="center"> Fig. 4  Visualization of feature vectors of LR face images from an untrained LRFECNN</p>
<hr style="border:2px solid gray"> </hr>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976863-0e76553b-3b76-441c-b181-95dbb3ce4bdd.png">
</p>
<p align="center"> Fig. 5  Visualization of feature vectors of HR face images from HRFECNN</p>  

## Training LRFECNN
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132976987-ccb5b7ef-02b8-46ca-a5e3-6e88ced756ae.png">
</p>
<p align="center"> Fig. 6  Variation of training loss and validation loss by training LRFECNN on 24 x 24 images</p>  
<hr style="border:2px solid gray"> </hr>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977096-3182dd10-2053-47ae-8c83-309fc49d1b1f.png">
</p>
<p align="center"> Fig. 7  Variation of recognition accuracy of the Logistic Regression classifier on 24 x 24 images </p>

##  PCA visualization post-training
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977050-4839d22f-532c-493c-b82c-3616942c76a4.png">
</p>
<p align="center"> Fig. 8  Visualization of feature vectors of LR probe images from LRFECNN post-training</p>

##  Evaluation on different probe resolutions and subjects in test set 1
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977180-1927a37c-b42d-42ea-bf62-5281a61eb000.png">
</p>
<p align="center"> Table 2 </p>

## Evaluation on different probe resolutions and subjects in test set 2
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977218-b473ab6e-49df-4ff0-933d-c6e046a2c84c.png">
</p>
<p align="center"> Table 3 </p>

## Evaluation on different super resolution methods
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977303-c083e008-7926-413a-8550-7d98e2d06c12.png">
</p>
<p align="center"> Fig. 9 Configurations with different super resolution methods. Blocks with
blue color are involved in the training phase. Super resolution via a) Bicubic
Interpolation, b) EDSR [6], c) WDSR [7], d) SRGAN [8] </p>  
<hr style="border:2px solid gray"> </hr>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977323-866e3dec-03be-4419-b511-20d447dee05a.png">
</p>
<p align="center"> Table 4 </p>  
<hr style="border:2px solid gray"> </hr>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68325029/132977339-39524c42-dff0-4b05-9db9-7f2241430c21.png">
</p>
<p align="center"> Fig. 10  Variation of recognition accuracy with different super resolution methods</p>

# References
1. E. Zangeneh, M. Rahmati, and Y. Mohsenzadeh, “Low resolution face
recognition using a two-branch deep convolutional neural network
architecture,” Expert Systems with Applications, vol. 139, p. 112854, 2020.
2. F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2015, pp. 815–823.
3. H. Taniai, “keras-facenet,” 2018. [Online]. Available: https://github.com/nyoki-mtl/keras-facenet
4. Astawa, I Nyoman Gede Arya (2020), “KomNET: Face Image Dataset from Various Media”, Mendeley Data, V2, doi:10.17632/hsv83m5zbb.2
5. A. Rosenfeld, M. D. Solbach, and J. K. Tsotsos, “Totally looks like-how
humans compare, compared to machines,” in Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition Workshops,
2018, pp. 1961–1964.
6. B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, “Enhanced deep
residual networks for single image super-resolution,” in Proceedings
of the IEEE conference on computer vision and pattern recognition
workshops, 2017, pp. 136–144.
7. J. Yu, Y. Fan, J. Yang, N. Xu, Z. Wang, X. Wang, and T. Huang,
“Wide activation for efficient and accurate image super-resolution,”
arXiv preprint arXiv:1808.08718, 2018.
8. C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, ´
A. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single
image super-resolution using a generative adversarial network,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2017, pp. 4681–4690.
