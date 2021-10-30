from numpy import hstack
from numpy import zeros
from numpy import ones
import numpy as np
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import math
import matplotlib.pyplot as plt
# define the standalone discriminator model
def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
	model = Sequential()
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(n_outputs, activation='linear'))
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
 
# generate n real samples with class labels
def generate_real_samples(n):
	# generate inputs in [-0.5, 0.5]
	X1 = rand(n)*2*np.pi
	# generate outputs X^2
	X2 = np.exp(0.5*X1)*np.sin(20*X1)
	# stack arrays
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = ones((n, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y
 
# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=1000):
    # prepare real samples
    x_real, y_real = generate_real_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    '''
    plt.figure(dpi=200)
    plt.clf()
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.savefig('./result/{}.png'.format(epoch))
    '''
 
# train the generator and discriminator
g_loss_curve = []
d_loss_curve = []

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        loss_d_real = d_model.train_on_batch(x_real, y_real)
        loss_d_fake = d_model.train_on_batch(x_fake, y_fake)
        loss_d = loss_d_real + loss_d_fake
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        loss_g = gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        # d_real, _ = d_model.evaluate(x_real, y_real)
        # d_fake, _ = d_model.evaluate(x_fake, y_fake)
        # g_loss, _ = g_model.evaluate(x_gan, y_gan)
        g_loss_curve.append(np.mean(loss_g))
        d_loss_curve.append(np.mean(loss_d))

        if (i+1) % n_eval == 0:
        	summarize_performance(i, g_model, d_model, latent_dim)
	
    plt.figure(dpi=200)
    plt.clf()
    plt.plot(g_loss_curve, label = 'g_loss_curve')
    plt.plot(d_loss_curve, label = 'd_loss_curve')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('./result/loss_curve1.png')


if __name__ == '__main__':
# size of the latent space
    latent_dim = 5
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    # train model

    train(generator, discriminator, gan_model, latent_dim)