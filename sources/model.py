import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from keras.activations import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K


def VGG16(input_size=(128, 128, 1), feature_maps=64, lr=1e-4, loss='binary_crossentropy', metric='accuracy'):

    inputs = Input(input_size)

    conv1 = Conv2D(feature_maps, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(feature_maps * 2, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(feature_maps * 4, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv3')(pool2)
    conv4 = Conv2D(feature_maps * 4, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv4')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(feature_maps * 8, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv5')(pool3)
    conv6 = Conv2D(feature_maps * 8, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv6')(conv5)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(feature_maps * 8, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv7')(pool4)
    conv8 = Conv2D(feature_maps * 8, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv8')(conv7)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv8)

    drop = Dropout(0.5)(pool5)

    flat = Flatten()(drop)

    dense1 = Dense(feature_maps * 64, activation='relu')(flat)

    dense2 = Dense(feature_maps * 64, activation='relu')(dense1)

    dense3 = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=inputs, outputs=dense3)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[metric])
    model.summary()

    return model

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def classifier(input_size=(128, 128, 1), feature_maps=64, lr=1e-4, loss='binary_crossentropy', metric='accuracy'):

    inputs = Input(input_size)

    conv1 = Conv2D(feature_maps, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(feature_maps * 2, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(feature_maps * 4, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv3')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(feature_maps * 4, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='conv4')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop1 = Dropout(0.3)(pool4)

    flat = Flatten()(drop1)

    dense1 = Dense(feature_maps * 8, activation='relu')(flat)

    dense2 = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[metric])
    model.summary()

    return model


