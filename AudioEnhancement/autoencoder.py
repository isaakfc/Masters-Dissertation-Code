from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow import keras

class Encoder(keras.layers.Layer):
    def __init__(self, filters, l2_reg):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0],
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.conv2 = Conv2D(filters=filters[1],
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.pool = MaxPooling2D((2, 2), padding='same')
        self.relu = keras.activations.relu
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.additional_conv = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding='same')

    def call(self, input_tensor, training=False):
        x1 = self.conv1(input_tensor)
        x1 = self.batch_norm1(x1, training=training)
        x1 = self.relu(x1)
        x1 = self.additional_conv(x1)
        x1_pooled = self.pool(x1)
        x2 = self.conv2(x1_pooled)
        x2 = self.batch_norm2(x2, training=training)
        x2 = self.relu(x2)
        x2_pooled = self.pool(x2)
        return x2_pooled, [x1, x2]
class Decoder(keras.layers.Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[1], kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding='same')
        self.conv3 = Conv2D(1, 3, 1, activation='sigmoid', padding='same')
        self.additional_conv = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding='same')
        self.upsample = UpSampling2D((2, 2))
        self.relu = keras.activations.relu

    def call(self, inputs):
        encoded, skip_features = inputs
        x1, x2 = skip_features
        x = self.conv1(encoded)
        x = self.relu(x)
        x = self.upsample(x)
        x = Concatenate()([x, x2])
        x = self.additional_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = Concatenate()([x, x1])

        return self.conv3(x)

class Autoencoder(keras.Model):
    def __init__(self, filters, l2_reg):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters, l2_reg)
        self.decoder = Decoder(filters)

    def call(self, input_features, training=False):
        encoded, skip_features = self.encoder(input_features, training=training)
        reconstructed = self.decoder((encoded, skip_features))
        return reconstructed