import tensorflow.keras.models as models
import tensorflow.keras.layers as layers


def conv(n_filters):
    return layers.Conv2D(n_filters, 3, activation='elu', padding='same', kernel_initializer='he_normal')


def conv_block(x, n_filters, batch_norm=True, dropout=0):
        y = conv(n_filters)(x)
        if batch_norm:
            y = layers.BatchNormalization()(y)
        y = conv(n_filters)(y)
        if batch_norm:
            y = layers.BatchNormalization()(y)
        if dropout:
            y = layers.Dropout(dropout)(y)
        return y


def up_sampling_block(x, n_filters, batch_norm=True):
        y = layers.UpSampling2D(size=(2, 2))(x)
        y = conv(n_filters)(y)
        if batch_norm:
            y = layers.BatchNormalization()(y)
        return y
        

def unet(input_size, n_classes, batch_norm=True, dropout=0):
    inputs = layers.Input(input_size)

    conv1 = conv_block(inputs, 64, batch_norm=batch_norm, dropout=dropout)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128, batch_norm=batch_norm, dropout=dropout)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256, batch_norm=batch_norm, dropout=dropout)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512, batch_norm=batch_norm, dropout=dropout)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 1024, batch_norm=batch_norm, dropout=dropout)

    up1 = up_sampling_block(conv5, 512)
    merge1 = layers.concatenate([conv4, up1])
    conv6 = conv_block(merge1, 512, batch_norm=batch_norm, dropout=dropout)

    up2 = up_sampling_block(conv6, 256)
    merge2 = layers.concatenate([conv3, up2])
    conv7 = conv_block(merge2, 256, batch_norm=batch_norm, dropout=dropout)

    up3 = up_sampling_block(conv7, 128)
    merge3 = layers.concatenate([conv2, up3])
    conv8 = conv_block(merge3, 128, batch_norm=batch_norm, dropout=dropout)

    up4 = up_sampling_block(conv8, 64)
    merge4 = layers.concatenate([conv1, up4])
    conv9 = conv_block(merge4, 64, batch_norm=batch_norm, dropout=dropout)

    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

