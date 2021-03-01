import tensorflow as tf
import tensorflow.keras as tfk

def define_cnn(n_doms):

    cnn_model = tfk.Sequential()
    cnn_model.add(tfk.layers.Conv2D(16, kernel_size=3, activation='relu'))
    cnn_model.add(tfk.layers.BatchNormalization())
    cnn_model.add(tfk.layers.MaxPool2D((2, 2)))
    cnn_model.add(tfk.layers.Conv2D(32, kernel_size=3, activation='relu'))
    cnn_model.add(tfk.layers.BatchNormalization())
    cnn_model.add(tfk.layers.MaxPool2D((2, 2)))
    cnn_model.add(tfk.layers.Conv2D(64, kernel_size=3, activation='relu'))
    cnn_model.add(tfk.layers.BatchNormalization())
    cnn_model.add(tfk.layers.MaxPool2D((2, 2)))
    cnn_model.add(tfk.layers.Conv2D(128, kernel_size=3, activation='relu'))
    cnn_model.add(tfk.layers.BatchNormalization())
    cnn_model.add(tfk.layers.MaxPool2D((2, 2)))
    cnn_model.add(tfk.layers.Flatten())
    cnn_model.add(tfk.layers.Dense(512, activation='relu'))
    cnn_model.add(tfk.layers.Dropout(0.5))
    cnn_model.add(tfk.layers.Dense(64, activation='relu'))
    cnn_model.add(tfk.layers.Dropout(0.5))
    cnn_model.add(tfk.layers.Dense(n_doms, activation=None))

    return cnn_model


    
def define_dual_cnn(in_dim1, in_dim2, n_doms):

    inputx = tfk.Input(shape=(in_dim1[0], in_dim1[1], in_dim1[2]), name='in_x')
    inputy = tfk.Input(shape=(in_dim2[0], in_dim2[1], in_dim2[2]), name='in_y')

    x = tfk.layers.Conv2D(32, kernel_size=3, padding='SAME')(inputx)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.ReLU()(x)
    x = tfk.layers.MaxPool2D((2, 2))(x)
    x = tfk.layers.Conv2D(64, kernel_size=3, padding='SAME')(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.ReLU()(x)
    x = tfk.layers.MaxPool2D((2, 2))(x)
    x = tfk.layers.Conv2D(128, kernel_size=3, padding='SAME')(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.ReLU()(x)
    x = tfk.layers.MaxPool2D((2, 2))(x)
    x = tfk.layers.Conv2D(256, kernel_size=3, padding='SAME')(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.ReLU()(x)
    x = tfk.layers.MaxPool2D((2, 2))(x)

    x = tfk.layers.Concatenate()([x,inputy])
    
    x = tfk.layers.Conv2D(512, kernel_size=3, padding='SAME')(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.ReLU()(x)

    x = tfk.layers.Flatten()(x)
    x = tfk.layers.Dense(1024, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    x = tfk.layers.Dense(128, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    output = tfk.layers.Dense(n_doms, activation=None)(x)


    return tfk.Model(inputs = [inputx, inputy], outputs = [output], name='base_cnn')