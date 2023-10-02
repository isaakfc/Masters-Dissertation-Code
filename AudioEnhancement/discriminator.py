from keras.layers import Dense, Conv2D
from keras.layers import Flatten, LeakyReLU
from tensorflow import keras


class Discriminator(keras.Model):
    def __init__(self, alpha):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=3, strides=2, padding="same")
        self.conv2 = Conv2D(128, kernel_size=3, strides=2, padding="same")
        self.conv3 = Conv2D(256, kernel_size=3, strides=2, padding="same")
        self.conv4 = Conv2D(512, kernel_size=3, strides=2, padding="same")
        self.leaky_relu = LeakyReLU(alpha)
        self.dense = Dense(1)  # No activation
        self.flatten = Flatten()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.flatten(x)
        return self.dense(x)