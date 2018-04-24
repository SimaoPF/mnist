'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.utils import plot_model
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import time

batch_size = 128
num_classes = 10
epochs = 8
str_name = 'Mnist_gen_mod1_Adam_'+ time.asctime()
epoch_iterations = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
data_gen_args = dict(featurewise_center=True, samplewise_center=True,
                     rotation_range=10.0, width_shift_range=0.05,
                     height_shift_range=0.05, shear_range=0.05,
                     zoom_range=[.8, 1.05], fill_mode='nearest',
                     validation_split=0.2)

Igen = ImageDataGenerator(**data_gen_args)
Igen.fit(x_train, augment=True)
xy_gen_t =Igen.flow(x_train,y_train,batch_size=batch_size,seed=1,subset="training")                          
xy_gen_v =Igen.flow(x_train,y_train,batch_size=batch_size/4,seed=1,subset="validation")                          

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('tanh'))
#
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adam(),
#              metrics=['accuracy','mae']);
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.binary_accuracy,
                       keras.metrics.binary_crossentropy,
                       keras.metrics.categorical_accuracy]);
              
plot_model(model, to_file='model.png',show_shapes=True)
saver = callbacks.ModelCheckpoint('/home/sfaria/Python/mnist_models/'+str_name+ '_mnistMODEL.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)    

tb = callbacks.TensorBoard(log_dir='/home/sfaria/Python/logs/'+str_name,
                           histogram_freq=0, write_graph=False, write_images=True)

#model.fit(x_train[5000:50000], y_train[5000:50000],
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_train[0:5000], y_train[0:5000]),
#          callbacks=[tb])
model.fit_generator(xy_gen_t,
          steps_per_epoch=epoch_iterations,
          epochs=epochs,
          validation_data=xy_gen_v,
          validation_steps=epoch_iterations/5,
          callbacks=[tb, saver],
           use_multiprocessing=True)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
