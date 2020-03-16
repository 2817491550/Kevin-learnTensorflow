# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import  glob
import os


# %%
(train_x,train_y),(_,_)=tf.keras.datasets.fashion_mnist.load_data()
train_x.shape


# %%
train_x.dtype


# %%
train_x=(train_x.reshape(60000,28,28,1).astype('float32')-127.5)/127.5
train_x.shape


# %%
BATCH_SIZE=256
BUFFER_SIZE=60000


# %%
dataset=tf.data.Dataset.from_tensor_slices(train_x,)
dataset


# %%
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset


# %%
def geneator_model():
    model=keras.Sequential([
        layers.Dense(256,input_shape=(100,),use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(512,use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(28*28*1,use_bias=False,activation='tanh'),
        layers.BatchNormalization(),
        #layers.Reshape(28,28,1)
        layers.Reshape((28,28,1))
])
    return model 


# %%
def discriminator_model():
    model=keras.Sequential([
        layers.Flatten(),
        layers.Dense(512,use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(256,use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(1) ])
    return model


# %%
cross_entropy=keras.losses.BinaryCrossentropy(from_logits=True)


# %%
def discriminator_loss(real_output,fake_output):
    real_loss=cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
    return real_loss+fake_loss


# %%
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output),fake_output)


# %%
generator_opt=keras.optimizers.Adam(1e-4)
discriminator_opt=keras.optimizers.Adam(1e-4)


# %%
EPOCHS=100
noise_dim=100
num_example_generate=16
seed=tf.random.normal([num_example_generate,noise_dim])


# %%
geneator=geneator_model()

discriminator=discriminator_model()


# %%



# %%
def train_step(images):
    noise=tf.random.normal([BATCH_SIZE,noise_dim])
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        real_output=discriminator(images,training=True)
        gen_image=geneator(noise,training=True)
        fake_output=discriminator(gen_image,training=True)
        gen_loss=generator_loss(fake_output)
        disc_loss=discriminator_loss(real_output,fake_output)

    gradient_gen=gen_tape.gradient(gen_loss,geneator.trainable_variables)
    gradient_disc=disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen,geneator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc,discriminator.trainable_variables))


# %%
def generate_plt_images(gen_model,test_noise):
    pre_images=gen_model(test_noise,training=False)
    fig=plt.figure(figsize=(4,4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(pre_images[i,:,:,0],cmap='gray')
        plt.axis('off')
    plt.show()


# %%
def train(dataset,epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
            print('.',end='')
        generate_plt_images(geneator,seed)


# %%
train(dataset,EPOCHS)


# %%


