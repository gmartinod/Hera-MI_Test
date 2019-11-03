# HERA-MI, Machine Learning Ingeneer
## Pre-Interview entrance test
## Guillaume Martinod, 03/11/2019

### 1. Dataset preparation

#### data.find_dicom()
From the repository downloaded on GitHub, I searched for every Dicom Data pathes and splited it into a list of images and a list of corresponding masks 

#### data.read_adjust_dicom()
Then, for each of the 4 cases (image and mask), I loaded the Dicoms as numpy arraies and pre-process it. In 3 cases, the image and the mask have the same size, but the last one has a smaller mask. Without more info about the relative localization of both, I cropped each side of the image to fit the mask dimensions.
I added a step of normalization between percentil 2 and percentil 99 to bring the range of pixel values between 0 and 1 (with precision float32). To handle the data easier I resized every image and mask to size (2200, 1200) corresponding to 40% of the mean size (5500, 3000).
I combined each image with its corresponding mask in a 3D array, stored in a list. 

#### data.create_patches()
From each case resized and normalized, I extracted 1000 (128, 128) patches with extract_patches_2d() from sklearn library. To avoid learning on patches full of zeros, I removed the patches for which the sum of the pixel values is lower than a threshold.
Each non empty patch is considered as "Mass" (classe 1) if its corresponding mask has 5% of positive pixels. Otherwise the patch is considered "Non mass" (classe 0). This reference annotation is stored in a 1D binary array.
The patches and annotations are stored in 2 lists, and the percentage of masses in the dataset is calculated (average of 7% of mass patches)

#### data.two_classes_split()
The next step is to differenciate the mass and non mass patches in two different lists. We get a list of the 4 cases, each composed of a list of the mass patches and a list of the non mass patches. 

### 2. Data augmentation and dataset split

#### augmentation.Augmentation()
The first step of the data augmentation is to obtain a balanced dataset with 50% of mass patches and 50% of non mass patches.
For each case, I chose to concatenate the mass patches with the same number of non mass patches. The corresponding 1D annotation array is composed of a first half of 1 (representing mass class), and the second one of 0 (represneting non mass class)
Then I build the train, validation and test split. To keep a patient-wise coherence and avoid possible overlap between the 3 splits, I chose to divide the database in 50%, 25% and 25% : the train set is build from case 0 and 1 (around 175 patches), validation set from case 2 (around 60 patches), and test set from case 3 (around 80 patches). To fit keras' standard, I add one dimension to x and y array of each set. 
Then, I set the range of each parameter of the ImageDataGenerator() function of Keras. The deformations chosen are the ones that seemed reasonnable to me, looking at the resulting images. The range of each parameter is tuned to improve the classification results. Because it caracterise the learning process, validation and test sets need to represent real patients and does not require data augmentation. 
Every set is passed by a generator by batches of 30 patches. 

### 3. Network architecture : classifier based on VGG16

#### model.VGG16()
I based the learning process on VGG16 architecture descibed on https://neurohive.io/en/popular-networks/vgg16/ (configuration A). I chose this network because of its popularity for classification tasks and because it is easy to train end-to-end on CPU (59,5 M of trainable parameters and one hour for the trainning process). I added a Drop-Out layer to improve generalization during inference. 
The number of feature maps is 64 for the first convolutionnal layer and double for each max pooling to reach 512. The learning rate is set to 5e-5 with Adam optimizer to have the best speed of convergence. As we work on balanced datasets, I chose the binary cross-entropy loss and the accuracy metric from Keras library. These loss and metric doesn't work for highly imbalanced datasets, but work best in our case. The number of epoch is set to 100, enough to let the optimization converge.

#### train.trainning_session()
Using Keras ModelCheckpoint() callback, I chose to maximize the validation accuracy during the optimization process. The weights are updated only if the they improve the accuracy on the validation set. 
The EarlyStopping() callback is used to stop the trainning session if the weights were not update for more epochs than the patience (set to 20 epochs).

#### test.testing_session()
When the trainning sessing is done, the model is loaded with the last weights updated. I run model.predict() on the test set with a batch size of 1 patch, to evaluate the probability of each patch to be a mass. This probability is thresholded at 0.5 to determine the predicted class of each patch. 

I used classification_report() from sklearn.metrics to compare the predicted result on test set, with the reference. From the few optimizations I ran with this configuration, I got a mean accuracy of 63%. This result is not enough to differenciate the mass from the non mass in a robust way. Eventhough, the learning process is going well, as we can see on the minimization of the loss function. We have to keep in mind that the network is trained with only 2 full mammographs, making it hard to objectively caracterise a mass. 

### 4. On run metrics visualization

#### train.trainning_session()
To visualize the on going trainning session, I used the Tensorboard tool provided by Tensorflow. It is implemented as a Keras callback and plot to the value of the loss and metric of train and validation sets at the end of each epoch. It is locally run during the trainning and the graphs are accessible on http://localhost:6006/ . To be run in a Virtualenv, Tensorboard must be launch from program.Tensorboard(). 

### Additionnal tests and content

During the implementation of this solution, I tried different configuration to propose the best one yet. I have left in the code some trace of what I tried : 

#### model.classifier()
I originally implemented a shallower network, easier to train (3 M parameters). Tested with different parameters, it was really bad to differenciate mass from non mass. I tried the deeper network VGG16 and it improved the results. 

#### train.trainning_session()
Before balancing the database in the pre-processing process, I kept all the extracted patches, and the database was composed of 93% of non mass patches for 7% of mass patches. To overcome this imbalance, I tried to weight each class propostionnaly (class_weight parameter in model.fit_generator()). It doesn't improve the results, maybe because the imbalance was too big. 

#### model.recall_m()
Also in an imbalanced configuration, I tried to maximize the recall metric during optimization process. It would have permited to focus on the prediction of the mass patches, without taking into account the great majority of the non mass patches. It doesn't help in the trainning process because the metric is calculated batch-wise and isn't representative of the prediction on each patch. 


### Virtualenv configuration and libraries : 

Python version 3.6.6

pip list : 
Package              Version
-------------------- --------
absl-py              0.8.1
astor                0.8.0
cycler               0.10.0
gast                 0.2.2
google-pasta         0.1.7
grpcio               1.24.3
h5py                 2.10.0
joblib               0.14.0
Keras                2.3.1
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
kiwisolver           1.1.0
Markdown             3.1.1
matplotlib           3.1.1
numpy                1.17.3
opencv-python        4.1.1.26
opt-einsum           3.1.0
Pillow               6.2.1
pip                  10.0.1
protobuf             3.10.0
pydicom              1.3.0
pyparsing            2.4.2
python-dateutil      2.8.0
PyYAML               5.1.2
scikit-learn         0.21.3
scipy                1.3.1
setuptools           39.1.0
six                  1.12.0
sklearn              0.0
tensorboard          2.0.0
tensorflow           2.0.0
tensorflow-estimator 2.0.1
termcolor            1.1.0
Werkzeug             0.16.0
wheel                0.33.6
wrapt                1.11.2

