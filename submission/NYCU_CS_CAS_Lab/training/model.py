import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, DepthwiseConv2D

tf.random.set_seed(0)

def model_ds_max5():
    input_shape = [1, 1250, 1]
    inputs = Input(shape=input_shape)
    # MaxPooling
    x = MaxPooling2D(pool_size=(1,5), strides=(1,5))(inputs)
    # Conv1
    x = Conv2D(3, kernel_size=(1,6), strides=(1,2), padding="same")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # Ds Conv1
    x = DepthwiseConv2D(kernel_size=(1,5), strides=(2,2), depth_multiplier=1, padding="same")(x) #in:3 out:3
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # Ds Conv2
    x = DepthwiseConv2D(kernel_size=(1,4), strides=(1,1), depth_multiplier=2, padding="same")(x) #in:3 out:6
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(1,3), strides=(1,2), padding="same")(x)
    # Ds Conv3
    x = DepthwiseConv2D(kernel_size=(1,4), strides=(2,2), depth_multiplier=2, padding="same")(x) #in:6 out:12
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # Ds Conv4
    x = DepthwiseConv2D(kernel_size=(1,4), strides=(3,3), depth_multiplier=2, padding="same")(x) #in:12 out:24
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # FC
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

