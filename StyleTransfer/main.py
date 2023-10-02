import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import librosa
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

# Original Repository Link: https://github.com/inzva/Audio-Style-Transfer

CONTENT_FILENAME = ""
STYLE_FILENAME = ""
N_FFT = 2048

def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    x = x / np.max(np.abs(x))
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S[:, :430]))
    return S, fs

a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]

N_FILTERS = 4096

a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std

g = tf.Graph()

with g.as_default(), g.device('/cpu:0'), tf.compat.v1.Session() as sess:
    x = tf.compat.v1.placeholder('float32', [1, 1, N_SAMPLES, N_CHANNELS], name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")

    net = tf.nn.relu(conv)

    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})

    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES

from sys import stderr

learning_rate = 2e-1
ALPHA = 1e-2
BETA = 2
iterations = 400

result = None
with tf.compat.v1.Graph().as_default():
    x = tf.compat.v1.Variable(np.random.randn(1, 1, N_SAMPLES, N_CHANNELS).astype(np.float32) * 1e-3, name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")

    net = tf.nn.relu(conv)

    content_loss = ALPHA * 2 * tf.nn.l2_loss(
        net - content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
    style_loss = BETA * tf.nn.l2_loss(gram - style_gram)

    loss = content_loss + style_loss

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        print('Started optimization.')
        for i in range(iterations):
            sess.run(train_op)
            if i % 10 == 0:
                print(f"Iteration {i}, loss: {sess.run(loss)}")

        print('Final loss:', sess.run(loss))
        result = sess.run(x)

a = np.zeros_like(a_content)
a[:N_CHANNELS, :] = np.exp(result[0, 0].T) - 1

p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, n_fft=N_FFT))


vmin = np.min(a_content)
vmax = np.max(a_content)

plt.figure(figsize=(15, 5))

# For the first subplot
plt.subplot(1, 3, 1)
plt.title('Sample Based Passage (Content)')
plt.imshow(a_content[:400,:], vmin=vmin, vmax=vmax)

# For the second subplot
plt.subplot(1, 3, 2)
plt.title('Recorded Passage Same Style (Style)')
plt.imshow(a_style[:400,:], vmin=vmin, vmax=vmax)

# For the third subplot
plt.subplot(1, 3, 3)
plt.title('Style Transfered Output')
plt.imshow(a[:400,:], vmin=vmin, vmax=vmax)

plt.savefig('outputs/content_style_and_reconstructed_spectrogram.png')
plt.close()

sf.write("outputs/STYLE_TRANSFERRED_AUDIO.wav", x, fs)