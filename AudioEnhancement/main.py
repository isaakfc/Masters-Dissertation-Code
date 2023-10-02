import load
import tensorflow as tf
import autoencoder
import discriminator
import plot
import train
import numpy as np
import audiogeneration


# =========================
# DATA PREPARATION
# =========================

# Paths for data
SPECTROGRAMS_PASSAGES_PATH = ""
SPECTROGRAMS_RECREATIONS_PATH = ""
GENUINE_RECORDINGS_PATH = ""

# Load spectrograms
x_clean = load.load_data_set(SPECTROGRAMS_PASSAGES_PATH)
x_noisy = load.load_data_set(SPECTROGRAMS_RECREATIONS_PATH)
x_genuine = load.load_data_set(GENUINE_RECORDINGS_PATH)
x_genuine = np.concatenate((x_genuine, x_clean), axis=0)

print(x_clean.shape)

# Convert to 32 bit floating point
x_clean = x_clean.astype('float32')
x_noisy = x_noisy.astype('float32')
x_genuine = x_genuine.astype('float32')

# Normalise
overall_min = min(np.min(x_genuine), np.min(x_noisy))
overall_max = max(np.max(x_genuine), np.max(x_noisy))

x_clean = (x_clean - overall_min) / (overall_max - overall_min)
x_noisy = (x_noisy - overall_min) / (overall_max - overall_min)
x_genuine = (x_genuine - overall_min) / (overall_max - overall_min)

# Get the indicies from the training array, shuffle and limit to 50
indices = np.arange(x_clean.shape[0])
np.random.shuffle(indices)
test_indices = indices[:50]

# Take the corrosponding random indicies from the training arrays to make test set
x_clean_test = x_clean[test_indices]
x_noisy_test = x_noisy[test_indices]

# Remove the test data from training set
x_clean_train = np.delete(x_clean, test_indices, axis=0)
x_noisy_train = np.delete(x_noisy, test_indices, axis=0)

# =========================
# HYPER PARAMETERS
# =========================

GENERATOR_LEARNING_RATE = 1e-6
DISCRIMINATOR_LEARNING_RATE = 1e-6
MIN_LR = 0.00000001
DECAY_FACTOR = 1.00004
EPOCHS = 15
L2_REG = 0.01
BATCH_SIZE = 14
FILTERS = [64, 64, 32]
ALPHA = 0.1
BETA_1 = 0.75
GAMMA = 0.35
ZETA = 0.05
PRINT_FREQ = 5
LAMBDA = 22
N_CRITIC = 3

# =========================
# GLOBAL PARAMETERS
# =========================

SR = 22000
HOP_LENGTH = 256
FRAME_SIZE = 512

# =========================
# TRAINING SET-UP
# =========================

# Initialise models
generatorModel = autoencoder.Autoencoder(FILTERS, L2_REG)
discriminatorModel = discriminator.Discriminator(ALPHA)

# Initialise optimizers
generator_optimizer = tf.keras.optimizers.legacy.Adam(GENERATOR_LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1)

batch_count = int(x_clean_train.shape[0] / BATCH_SIZE)
print('Epochs:', EPOCHS)
print('Batch size:', BATCH_SIZE)
print('Batches per epoch:', batch_count)

# To be used for plot
dLosses = []
gTotalLosses = []
gLosses = []
gRecLosses = []
gPercepLosses = []

trace = True
n_critic_count = 0

# =========================
# MAIN TRAINING LOOP
# =========================

for e in range(1, EPOCHS+1):
    print('\n', '-' * 15, 'Epoch %d' % e, '-' * 15)

    # Adjust learning rates
    DISCRIMINATOR_LEARNING_RATE = train.learning_rate_decay(DISCRIMINATOR_LEARNING_RATE, DECAY_FACTOR, MIN_LR)
    GENERATOR_LEARNING_RATE = train.learning_rate_decay(GENERATOR_LEARNING_RATE, DECAY_FACTOR, MIN_LR)
    print('current_learning_rate_discriminator %.10f' % (DISCRIMINATOR_LEARNING_RATE,))
    print('current_learning_rate_generator %.10f' % (GENERATOR_LEARNING_RATE,))
    train.set_learning_rate(DISCRIMINATOR_LEARNING_RATE, discriminator_optimizer)
    train.set_learning_rate(GENERATOR_LEARNING_RATE, generator_optimizer)

    # Shuffle training data
    permutation = np.random.permutation(x_clean_train.shape[0])
    x_train_shuffled = x_clean_train[permutation]
    x_train_noisy_shuffled = x_noisy_train[permutation]
    x_genuine_shuffled = np.random.permutation(x_genuine)

    for batch_idx in range(batch_count):
        # Get start and end index
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        # Extract from shuffled data
        x_real_batch = x_train_shuffled[start_idx:end_idx]
        x_noisy_batch = x_train_noisy_shuffled[start_idx:end_idx]
        x_genuine_batch = x_genuine_shuffled[start_idx:end_idx]

        # Train discriminator model
        d_loss, generated_spectro = train.WGAN_GP_train_d_step(x_genuine_batch,
                                             x_noisy_batch,
                                             discriminatorModel,
                                             generatorModel,
                                             discriminator_optimizer,
                                             LAMBDA,
                                             BATCH_SIZE)
        n_critic_count += 1

        if n_critic_count >= N_CRITIC:
            # Get embeddings
            real_embeddings, fake_embeddings = train.get_audio_embeddings2(x_real_batch, generated_spectro)
            # Train generator
            total_g_loss, r_loss, g_loss, p_loss, generated_spectrograms = train.WGAN_GP_train_g_step(x_real_batch,
                                                                                              x_noisy_batch,
                                                                                              real_embeddings,
                                                                                              fake_embeddings,
                                                                                              discriminatorModel,
                                                                                              generatorModel,
                                                                                              generator_optimizer,
                                                                                              GAMMA,
                                                                                              ZETA)
            x_real_batch_for_plot = x_real_batch
            x_noisy_batch_for_plot = x_noisy_batch
            n_critic_count = 0

        if (batch_idx + 1) % PRINT_FREQ == 0:
            print(
            f"\rEpoch {e}, Batch {batch_idx + 1}/{batch_count}: Discriminator Loss: {d_loss:.4f}, /"
            f"Generator Loss: {g_loss:.4f}, / Reconstruction Loss: {r_loss:.4f},/ Perceptual Loss: {p_loss:.8f},"
            f"/ Total generator Loss: {total_g_loss:.4f}")

        print(f"\rEpoch {e}, Batch {batch_idx + 1}/{batch_count}")


    if e == 1 or e % 1 == 0:
        plot.saveGeneratedSpectrograms(e, generated_spectrograms, x_real_batch_for_plot, x_noisy_batch_for_plot)


    dLosses.append(d_loss)
    gLosses.append(g_loss)
    gRecLosses.append(r_loss)
    gTotalLosses.append(total_g_loss)
    gPercepLosses.append(p_loss)


plot.plot_loss(e, dLosses, gTotalLosses, gRecLosses, gLosses, p_loss)

# Start new line
print()


generated_test_samples = generatorModel.predict(x_noisy_test)

audiogeneration.convert_spectrograms_to_audio(e,
                                              generated_test_samples,
                                              overall_max,
                                              overall_min,
                                              HOP_LENGTH,
                                              FRAME_SIZE,
                                              SR,
                                              num_files=48,
                                              file_prefix="test_transformed",
                                              output_dir="TEST_AUDIO/TEST_TRANSFORMED_AUDIO")

audiogeneration.convert_spectrograms_to_audio(e,
                                              x_noisy_test,
                                              overall_max,
                                              overall_min,
                                              HOP_LENGTH,
                                              FRAME_SIZE,
                                              SR,
                                              num_files=48,
                                              file_prefix="test_input",
                                              output_dir="TEST_AUDIO/TEST_INPUT_AUDIO")

audiogeneration.convert_spectrograms_to_audio(e,
                                              x_clean_test,
                                              overall_max,
                                              overall_min,
                                              HOP_LENGTH,
                                              FRAME_SIZE,
                                              SR,
                                              num_files=48,
                                              file_prefix="test_target",
                                              output_dir="TEST_AUDIO/TEST_TARGET_AUDIO")