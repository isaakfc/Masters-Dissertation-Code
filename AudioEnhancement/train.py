import tensorflow as tf
from keras import backend as K
import openl3
import numpy as np

input_repr2, content_type2, embedding_size2 = 'mel256', 'music', 512
model = openl3.models.load_audio_embedding_model(input_repr2, content_type2, embedding_size2, frontend='librosa')


def reshape_and_pad_tensor2(input_tensor, target_shape=(256, 199)):
    b_size, height, width, num_channels = input_tensor.shape
    shaped_tensor_list = []

    for i in range(b_size):
        sample = input_tensor[i]
        segment_list = []

        for j in range(0, width, target_shape[1]):
            segmented = sample[:, j:j + target_shape[1], :]

            pad_width = target_shape[1] - segmented.shape[1]
            padding_amount = np.zeros((target_shape[0], pad_width, num_channels))
            segmented = np.concatenate([segmented, padding_amount], axis=1)

            segment_list.append(segmented)

        reshaped_tensor = np.stack(segment_list)
        shaped_tensor_list.append(reshaped_tensor)

    output_tensor = np.concatenate(shaped_tensor_list, axis=0)
    return output_tensor


def get_audio_embeddings2(x_real, fake_image):
    real_reshape_x = reshape_and_pad_tensor2(x_real)
    fake_image_reshape = reshape_and_pad_tensor2(fake_image)

    real_reshape_x_flat = real_reshape_x
    fake_reshape_image_flat = fake_image_reshape

    real_embbeded = model(real_reshape_x_flat, training=False)
    fake_embedded = model(fake_reshape_image_flat, training=False)

    return real_embbeded, fake_embedded




@tf.function
def WGAN_GP_train_d_step(real_x,
                         noisy_x,
                         discriminator,
                         generator,
                         discriminator_optimizer,
                         LAMBDA,
                         batch_size):

    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noisy_x, training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_x, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        fake_pred = discriminator(fake_image, training=True)
        real_pred = discriminator(real_x, training=True)

        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,
                                  discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    discriminator_optimizer.apply_gradients(zip(D_gradients,
                                    discriminator.trainable_variables))

    return D_loss, fake_image

@tf.function
def WGAN_GP_train_g_step(real_x, noisy_x, real_embeddings, fake_embeddings, discriminator, generator, generator_optimizer, gamma=0.5, zeta=0.25):
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        #Crtitic score for fake image
        fake_image = generator(noisy_x, training=True)
        fake_pred = discriminator(fake_image, training=True)

        # Compute perceptual loss
        perceptual_loss = tf.reduce_mean(tf.square(real_embeddings - fake_embeddings))
        # Compute reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(real_x - fake_image))  # MSE
        # Compute WGAN loss
        g_loss = -tf.reduce_mean(fake_pred)

        # Combined loss
        total_g_loss = g_loss + gamma * recon_loss + zeta * perceptual_loss

    # Calculate the gradients for generator
    g_gradients = g_tape.gradient(total_g_loss,
                                  generator.trainable_variables)
    # Apply the gradients to the optimizer
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return total_g_loss, recon_loss, g_loss, perceptual_loss, fake_image


def learning_rate_decay(current_lr, decay_factor, MIN_LR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, optimizer):
    '''
        Set new learning rate to optimizers
    '''
    K.set_value(optimizer.lr, new_lr)