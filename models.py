from keras.models import load_model, Model, Sequential
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding2D)
from keras.layers.advanced_activations import ELU

def VGG16(num_features=4096):
    # ========================================================================
    # VGG-16 ARCHITECTURE
    # ========================================================================
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_features, name='fc6',
                    kernel_initializer='glorot_uniform'))
    return model

def lstm(seq_length=40, feature_length=4096, nb_classes=1):
    """
    Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently.

    nb_classes is set to 1 by default since its a binary classification problem.
    """
    input_shape = (1, feature_length)
    model = Sequential()
    model.add(LSTM(feature_length, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def mlp(num_features, batch_norm):
    extracted_features = Input(shape=(num_features,),
                               dtype='float32', name='input')
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99,
                               epsilon=0.001)(extracted_features)
        x = Activation('relu')(x)
    else:
        x = ELU(alpha=1.0)(extracted_features)

    x = Dropout(0.9)(x)
    x = Dense(4096, name='fc2', kernel_initializer='glorot_uniform')(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
    else:
        x = ELU(alpha=1.0)(x)
    x = Dropout(0.8)(x)
    x = Dense(1, name='predictions',
              kernel_initializer='glorot_uniform')(x)
    x = Activation('sigmoid')(x)

    model = Model(input=extracted_features,
                  output=x, name='classifier')
    return model
