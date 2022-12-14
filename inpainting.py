import tensorflow as tf

print(tf.__version__)
import numpy as np

tf.enable_eager_execution()
import matplotlib.pyplot as plt
import os


# Construction generator and discriminator

# construction generator class

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully connected layer
        self.fc1 = tf.keras.layers.Dense(units=4 * 4 * 512, activation=None)

        # BN + ReLU
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation1 = tf.keras.layers.Activation(activation='relu')

        # transpose convolution layer 1 transpose convolution to shape: (batch_size, 8, 8, 256)
        self.transp_conv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding="SAME", activation=None)

        # BN + ReLU
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation2 = tf.keras.layers.Activation(activation='relu')

        # transpose convolution layer 2 transpose convolution to shape: (batch_size, 16, 16, 128)
        self.transp_conv2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding="SAME", activation=None)

        # BN + ReLU
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation3 = tf.keras.layers.Activation(activation='relu')

        # transpose convolution layer 3 transpose convolution to shape: (batch_size, 32, 32, 64)
        self.transp_conv3 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="SAME", activation=None)

        # BN + ReLU
        self.bn4 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation4 = tf.keras.layers.Activation(activation='relu')

        # transpose convolution layer 4 transpose convolution to shape: (batch_size, 64, 64, 3)
        self.transp_conv4 = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding="SAME", activation=None)
        self.out = tf.keras.layers.Activation(activation='tanh')

    # call functions allow classes to be called as functions
    def call(self, z, is_training):
        fc1 = self.fc1(z)
        fc1_reshaped = tf.reshape(fc1, (-1, 4, 4, 512))

        bn1 = self.bn1(fc1_reshaped, training=is_training)
        activation1 = self.activation1(bn1)

        trans_conv1 = self.transp_conv1(activation1)
        bn2 = self.bn2(trans_conv1, training=is_training)
        activation2 = self.activation2(bn2)

        transp_conv2 = self.transp_conv2(activation2)
        bn3 = self.bn3(transp_conv2, training=is_training)
        activation3 = self.activation3(bn3)

        transp_conv3 = self.transp_conv3(activation3)
        bn4 = self.bn4(transp_conv3, training=is_training)
        activation4 = self.activation4(bn4)

        transp_conv4 = self.transp_conv4(activation4)
        output = self.out(transp_conv4)

        return output


# construction Discriminator class
class Discriminator(tf.keras.Model):
    def __init__(self, alpha):
        super(Discriminator, self).__init__()
        # Convolution layer 1 convolution to shape: (batch_size, 32, 32, 64)
        self.conv1 = tf.keras.layers.Conv2D(64, 5, strides=2, padding='SAME', activation=None)
        self.activation1 = tf.keras.layers.LeakyReLU(alpha=alpha)

        # convolution layer 2 convolution to shape: (batch_size, 16, 16, 128)
        self.conv2 = tf.keras.layers.Conv2D(128, 5, strides=2, padding='SAME', activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation2 = tf.keras.layers.LeakyReLU(alpha=alpha)

        # convolution layer 3 convolution to shape: (batch_size, 8, 8, 256)
        self.conv3 = tf.keras.layers.Conv2D(256, 5, strides=2, padding='SAME', activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation3 = tf.keras.layers.LeakyReLU(alpha=alpha)

        # convolution layer 4 convolution to shape: (batch_size, 4, 4, 512)
        self.conv4 = tf.keras.layers.Conv2D(512, 5, strides=2, padding='SAME', activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation4 = tf.keras.layers.LeakyReLU(alpha=alpha)

        # Pull the input into a one-dimensional vector Convolution to shape: (batch_size*4*4*512)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=1, activation=None)
        self.out = tf.keras.layers.Activation(activation='sigmoid')

    def call(self, inputs, is_training):
        conv1 = self.conv1(inputs)
        activation1 = self.activation1(conv1)

        conv2 = self.conv2(activation1)
        bn1 = self.bn1(conv2, training=is_training)
        activation2 = self.activation2(bn1)

        conv3 = self.conv3(activation2)
        bn2 = self.bn2(conv3, training=is_training)
        activation3 = self.activation3(bn2)

        conv4 = self.conv4(activation3)
        bn3 = self.bn3(conv4, training=is_training)
        activation4 = self.activation4(bn3)

        flat = self.flatten(activation4)
        logits = self.fc1(flat)
        out = self.out(logits)
        return out, logits


# Input noise dimension
z_dim = 100
learning_rate = 0.0002
# slope of leakyRelu
alpha = 0.2
# Decay rate of Adm optimizer
beta1 = 0.5
smooth = 0.1

batch_size = 128

# train times
counter = 0
# epochs times
epoch = 10
# Crop the size of the image
image_size = 108
image_shape = [64, 64, 3]
# number of samples
sample_num = 64
generator_net = Generator()
discriminator_net = Discriminator(alpha=alpha)


# Define the cost functions
def generator_loss(d_logits_fake, d_model_fake):
    # Generate a cost function for the generator
    # Input fake image, confuse the discriminator to determine the approximation 1
    g_loss = tf.reduce_mean(
        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    return g_loss


def discriminator_loss(d_logits_real, d_logits_fake, smooth=0.1):
    # Discriminator two cost functions
    # Input true image, judge approximation 1
    d_loss_real = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                                      labels=tf.ones_like(
                                                                                          d_logits_real) * (
                                                                                                     1 - smooth)))

    # Enter a fake image and determine the approximation to 0
    d_loss_fake = tf.reduce_mean(
        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake
    return d_loss


# Define Optimizer
global_counter = tf.compat.v1.train.get_or_create_global_step()
generator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
# Processing data and data display
from glob import glob

# Data set
# # Get all image paths
datas = glob(os.path.join('data/img_align_celeba/', '*.jpg'))
datas = datas[:10000]

# Display images function
def display_images(dataset, figsize=(4, 4), denomalize=True):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=figsize, )
    for ii, ax in enumerate(axes.flatten()):
        img = dataset[ii, :, :, :]
        if denomalize:
            img = ((img + 1) * 255 / 2).astype(np.uint8)  # Scale back to 0-255

        ax.imshow(img)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


import scipy.misc
import numpy as np
from PIL import Image
import skimage
from glob import glob
import imageio


# Images handling functions
# image reader
def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def imread(path):
    return imageio.imread(path).astype(np.float)

# image writer
def save_images(images, image_path):
    print('save_images')
    for imgindex in range(images.shape[0]):
        imageio.imsave(image_path + str(imgindex) + '.jpg', images[imgindex])

# transform images size to (64*64*3)
def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.

# crop image around center
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return skimage.transform.resize(x[j:j + crop_h, i:i + crop_w], [resize_w, resize_w])


# def imsave(images, size, path):
#   return scipy.misc.imsave(path, merge(images, size))


def convert_to_lower_resolution():
    images = glob(os.path.join('data/train', '*.jpg'))
    i = 0
    size = 108, 108
    for image in images:
        im = Image.open(image)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save("data/train" + str(i) + '.jpg')


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def inverse_transform(images):
    return (images + 1.) / 2.


# Run the model
# Generate test noise input
fake_input_test = tf.random.uniform(shape=(sample_num, z_dim),
                                    minval=-1.0, maxval=1.0, dtype=tf.float32)

num_batch = (int)(len(datas) / batch_size)
temp = 1
for i in range(epoch):
    # Randomly scrambled data
    np.random.shuffle(datas)
    for ii in range(num_batch):
        # Generate random noise in batches
        fake_input = tf.random.uniform(shape=(batch_size, z_dim),
                                       minval=-1.0, maxval=1.0, dtype=tf.float32)

        # Because the data is too large, the practice of processing the data set in batches
        batch_files = datas[ii * batch_size:(ii + 1) * batch_size]
        batch = [get_image(batch_file, image_size, is_crop=True) for batch_file in batch_files]
        batch_images = np.reshape(np.array(batch).astype(np.float32), [batch_size] + image_shape)

        with tf.GradientTape(persistent=True) as tape:

            # Run the generator
            g_model = generator_net(fake_input, is_training=True)

            # Input the real image to run the discriminator
            d_model_real, d_logits_real = discriminator_net(batch_images, is_training=True)

            # Enter a fake image to run the discriminator
            d_model_fake, d_logits_fake = discriminator_net(g_model, is_training=True)

            # Calculate the loss of the generator
            gen_loss = generator_loss(d_logits_fake, d_model_fake)

            # Calculate the loss of the discriminator
            dis_loss = discriminator_loss(d_logits_real, d_logits_fake, smooth)

            if counter % 1000 == 0:
                generated_samples = generator_net(fake_input_test, is_training=False)
                display_images(generated_samples.numpy())

            # Calculate the gradient for the variable
            discriminator_grads = tape.gradient(dis_loss, discriminator_net.variables)
            generator_grads = tape.gradient(gen_loss, generator_net.variables)

            # Perform gradient updates
            discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_net.variables),
                                                    global_step=global_counter)
            generator_optimizer.apply_gradients(zip(generator_grads, generator_net.variables),
                                                global_step=global_counter)
            generator_optimizer.apply_gradients(zip(generator_grads, generator_net.variables),
                                                global_step=global_counter)
            if i > 5:
                generator_optimizer.apply_gradients(zip(generator_grads, generator_net.variables),
                                                    global_step=global_counter)

            counter += 1
fake_input_test = tf.random.uniform(shape=(9, z_dim),
                                    minval=-1.0, maxval=1.0, dtype=tf.float32)
generated_samples = generator_net(fake_input_test, is_training=False)
display_images(generated_samples.numpy())
# Increase the sample set
sample_num = 64
sample_files = datas[0:sample_num]
sample = [get_image(sample_file, image_size, is_crop=True) for sample_file in sample_files]
sample_images = np.reshape(np.array(sample).astype(np.float32), [sample_num] + image_shape)


# Add image restoration cost function


def complete_Inpainting_loss(g_loss, mask, G, images, lam):
    # Loss of generation of the unbroken part of the real picture and the unbroken part of the fake picture
    contextual_loss = tf.reduce_sum(
        input_tensor=tf.contrib.layers.flatten(
            tf.abs(tf.multiply(mask, G) - tf.multiply(mask, images))), axis=1)
    # Perceived information loss (global structure guaranteed)
    perceptual_loss = g_loss
    complete_loss = contextual_loss + lam * perceptual_loss
    return complete_loss


# Generate MASK matrix


def generate_Mask(batch_size):
    # Percentage of obscured parts to all images
    scale = 0.25
    # MASK matrix for masked images
    mask = np.ones([batch_size] + image_shape).astype(np.float32)
    l = int(image_shape[0] * scale)
    u = int(image_shape[0] * (1.0 - scale))
    mask[:, l:u, l:u, :] = 0.0

    # Take out the MASK matrix of the broken part
    scale = 0.25
    imask = np.zeros([batch_size] + image_shape).astype(np.float32)
    l = int(image_shape[0] * scale)
    u = int(image_shape[0] * (1.0 - scale))
    imask[:, l:u, l:u, :] = 1.0

    return mask, imask


# Run model
if __name__ == '__main__':
    print("start")
    lam = 0.1
    # Parameters for training process
    nIndex = 500
    beta1 = 0.9
    beta2 = 0.9
    eps = 1e-9
    lr = 0.01
    batch_size = 64
    sample_mask, sample_imask = generate_Mask(sample_num)
    mask, imask = generate_Mask(batch_size)
    # Generate test noise input
    fake_input_test = tf.random.uniform(shape=(sample_num, z_dim),
                                        minval=-1.0, maxval=1.0, dtype=tf.float32)
    print("test")
    num_batch = (int)(len(datas) / batch_size)
    print("num_batch", num_batch)
    np.random.shuffle(datas)
    for i in range(num_batch):
        # Generate random noise in batches
        fake_input = tf.random.uniform(shape=(batch_size, z_dim),
                                       minval=-1.0, maxval=1.0, dtype=tf.float32)
        print("batch %d" % i)
        # Because the data is too large, the practice of processing the data set in batches
        batch_files = datas[i * batch_size:(i + 1) * batch_size]
        batch = [get_image(batch_file, image_size, is_crop=True) for batch_file in batch_files]
        batch_images = np.reshape(np.array(batch).astype(np.float32), [batch_size] + image_shape)

        m = 0
        v = 0
        for ii in range(nIndex):
            # print("batch: %d, index: %d" % (i, ii))
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(fake_input)
                # Run the generator
                g_model = generator_net(fake_input, is_training=False)

                # Enter a fake image to run the discriminator
                d_model_fake, d_logits_fake = discriminator_net(g_model, is_training=False)

                # Calculate the loss of the generator
                gen_loss = generator_loss(d_logits_fake, d_model_fake)
                complete_loss = complete_Inpainting_loss(gen_loss, mask, g_model, batch_images, lam)
                g = tape.gradient(target=complete_loss, sources=fake_input)

            if ii % 50 == 0:
                # Generate dummy images
                generated_samples = generator_net(fake_input, is_training=False)
                # Extract the broken part of the real image corresponding to the fake image
                fake_part = np.multiply(generated_samples, sample_imask)
                # Broken original image
                real_part = np.multiply(batch_images, sample_mask)
                # Channel stitching to get the original image
                inpainting_sample = np.add(fake_part, real_part)
                plt.subplot(121)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(generated_samples[0].numpy())
                plt.subplot(122)
                plt.imshow(inpainting_sample[0])
                plt.xticks([])
                plt.yticks([])

                plt.show()

            # Update for a single image (fake_input)
            m_prev = np.copy(m)
            v_prev = np.copy(v)

            m = beta1 * m_prev + (1 - beta1) * g[0]
            v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
            m_hat = m / (1 - beta1 ** (ii + 1))
            v_hat = v / (1 - beta2 ** (ii + 1))
            fake_input += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
            fake_input = tf.convert_to_tensor(value=np.clip(fake_input, -1, 1))

        counter += 1
