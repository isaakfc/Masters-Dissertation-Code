import matplotlib.pyplot as plt

def plot_loss(epoch, dLosses, gLosses, rLosses, standalone_gLosses, perceptual_loss):
    plt.figure(figsize=(10, 8))

    plt.plot(dLosses, label='Discriminative Loss', linewidth=2, color='magenta')

    plt.plot(gLosses, label='Total Generative Loss', linewidth=2, color='green')

    plt.plot(rLosses, label='Reconstruction Loss', color='orange', alpha=0.6)

    plt.plot(standalone_gLosses, label='Generator Loss', color='blue', alpha=0.6)

    plt.plot(perceptual_loss, label='Perceptual Loss', color='black', alpha=0.6)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('LOSSPLOTS/gan_loss_epoch_%d.png' % epoch)
    plt.close()

def saveGeneratedSpectrograms(epoch, generated_spectrograms, clean_spectrograms , noisy_spectrograms, dim=(3, 10), figsize=(20, 6)):


    num_to_plot = min(10, generated_spectrograms.shape[0])
    plt.figure(figsize=figsize)

    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(clean_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1 + dim[1])
        plt.imshow(noisy_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1 + 2*dim[1])
        plt.imshow(generated_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'SPECTROGRAMPLOTS/generated_image_epoch_{epoch}.png')
    plt.close()