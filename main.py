from sources.data import *
from sources.augmentation import *
from sources.train import *
from sources.test import *
import tensorflow as tf
from tensorflow import keras  # tf.keras
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


############################################################
# Main script leading each step of the end-to-end training #
############################################################

## Parameters :
Path_to_dicom = r"C:\Users\Utilisateur\Desktop\Hera-MI_test\test_material-master\test_material-master\task_03-machine_learning_engineer"
Resize_images = (2200, 1200)
Size_patch = (128, 128)
Nb_patch = 1000         # Number of patches generated from 1 image before selection of the non empty
Threshold_patch = 200   # Sum of the pixel values ([0,1]) mini to consider patch as non empty

weight_name = "tmp.hdf5"
weight_path = fr"C:\Users\Utilisateur\Desktop\Hera-MI_test\weights\{weight_name}"
logs_path = r"C:\Users\Utilisateur\Desktop\Hera-MI_test\logs"
feature_maps = 64
learning_rate = 5e-5
loss = 'binary_crossentropy'
metric = 'accuracy'
batch = 30
epochs = 100
patience = 20


## Patch Generation

lst_Image, lst_Mask = find_dicom(root=Path_to_dicom)

Dataset = read_adjust_dicom(ImPath=lst_Image,
                            Mask_Path=lst_Mask,
                            Resize=Resize_images)

Patches, Annotation, Proportion = create_patches(Dataset=Dataset,
                                                  Size=Size_patch,
                                                  Max_patches=Nb_patch,
                                                  Threshold=Threshold_patch)

diff_dataset = two_classes_split(dataset=Patches,
                                 annotation=Annotation)


## Data augmentation (parameter in function "Augmentation")

train_generator, valid_generator, test_generator, x_train, y_train, x_valid, y_valid, x_test, y_test = Augmentation(diff_dataset, batch)

## Training session

trainning_session(train_generator=train_generator,
                  valid_generator=valid_generator,
                  len_train=len(x_train),
                  len_valid=len(x_valid),
                  input_size=x_train[0].shape,
                  weight_path=weight_path,
                  logs_path=logs_path,
                  Proportion=0.5,
                  loss=loss,
                  metric=metric,
                  patience=patience,
                  feature_maps=feature_maps,
                  learning_rate=learning_rate,
                  batch=batch,
                  epochs=epochs)


## testing session

predicted = testing_session(weight_path=weight_path,
                             x_test=x_test,
                             y_test=y_test,
                             input_size=x_test[0].shape,
                             loss=loss,
                             metric=metric,
                             feature_maps=feature_maps,
                             learning_rate=learning_rate)


## print binary results

bi_results = predicted
bi_results[predicted > 0.5] = int(1)
bi_results[predicted <= 0.5] = int(0)

target_names = ['Non mass', 'Mass']
report = classification_report(y_test, bi_results, target_names=target_names)
matrice = confusion_matrix(y_test, bi_results)

print(f"Rapport de classification binaire : \n{report}")
print(f"Matrice de confusion binaire : \n{matrice}")

## print ROC and AUC

fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
