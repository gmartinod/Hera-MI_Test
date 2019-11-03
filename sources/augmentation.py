from tensorflow.keras.preprocessing.image import *
import numpy as np




def Augmentation (diff_dataset, batch=20):

    Patches = []
    Annotation = []
    for i, image in enumerate(diff_dataset):
        print(f"Shape of masses dataset in image {i} : {image[1].shape}")
        x = np.concatenate((image[1][:, :, :, 0], image[0][0:len(image[1]), :, :, 0]), axis=0)
        y = np.concatenate((np.ones(len(image[1])), np.zeros(len(image[1]))), axis=0)
        Patches.append(x)
        Annotation.append(y)

    print(f"Patches shape in image 0 : {Patches[0].shape}")

    # train set = image 0 and image 1
    x_train = np.concatenate((Patches[0][:, :, :], Patches[1][:, :, :]), axis=0)
    x_train = np.reshape(x_train, x_train.shape + (1,))
    y_train = np.concatenate((Annotation[0][:], Annotation[1][:]), axis=0)
    y_train = np.reshape(y_train, y_train.shape + (1,))
    print(f"x_train shape : {x_train.shape}; \ny_train shape : {y_train.shape}")

    # validation set = image 2
    x_valid = Patches[2][:, :, :]
    x_valid = np.reshape(x_valid, x_valid.shape + (1,))
    y_valid = Annotation[2][:]
    y_valid = np.reshape(y_valid, y_valid.shape + (1,))
    print(f"x_valid shape : {x_valid.shape}; \ny_valid shape : {y_valid.shape}")

    # test set = image 3
    x_test = Patches[3][:, :, :]
    x_test = np.reshape(x_test, x_test.shape + (1,))
    y_test = Annotation[3][:]
    y_test = np.reshape(y_test, y_test.shape + (1,))
    print(f"x_test shape : {x_test.shape}; \ny_test shape : {y_test.shape}")

    train_dataAug = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    valid_dataAug = ImageDataGenerator()
    test_dataAug = ImageDataGenerator()

    train_generator = train_dataAug.flow(x_train, y_train, batch_size=batch)
    valid_generator = valid_dataAug.flow(x_valid, y_valid, batch_size=batch)
    test_generator = test_dataAug.flow(x_test, y_test, batch_size=batch)

    return train_generator, valid_generator, test_generator, x_train, y_train, x_valid, y_valid, x_test, y_test
