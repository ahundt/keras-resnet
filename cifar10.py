"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
import datetime
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os


if K.image_dim_ordering() == 'tf':
    from keras.callbacks import TensorBoard
    import tensorflow as tf

import resnet

# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
    
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_sizes = [32]#[16,32,64]
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

out_dir=''
dirname=''

# Compile and train different models while meauring performance.
results = []
avg_time_results = []
for batch_size in batch_sizes:

    if K.image_dim_ordering() == 'tf':
        sess = tf.Session()
        from keras import backend as K
        K.set_session(sess)

    dirname = timeStamped(str(batch_size) + 'batch_cifar10_resnet')
    out_dir=dirname+'/'
    
    print('Running a new session in : ' + out_dir)
    
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # TODO switch to ResnetBuilder.build calls, permute parameters in 3x3 grid, generate plots for final assignment, save models?
    # use https://github.com/fchollet/keras/blob/master/examples/lstm_benchmark.py
    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, csv_logger])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        csv = CSVLogger(out_dir+dirname+'.csv', separator=',', append=True)
        model_checkpoint = ModelCheckpoint(out_dir+'weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks=[lr_reducer, early_stopper, csv]
        
        if K.image_dim_ordering() == 'tf':
            tensorboard = TensorBoard(log_dir=out_dir, histogram_freq=10, write_graph=True)
            callbacks.append(tensorboard)

        start_time = time.time()
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            verbose=1, max_q_size=100,
                            callbacks=callbacks)

        end_fit_time = time.time()
        average_time_per_epoch = (end_fit_time - start_time) / nb_epoch
        
        model.predict(X_test, batch_size=batch_size, verbose=1)

        end_predict_time = time.time()
        average_time_to_predict = (end_predict_time - end_fit_time) / nb_epoch

        results.append((history, average_time_per_epoch, average_time_to_predict))
        print ('--------------------------------------------------------------------')
        print ('[run_name,batch_size,average_time_per_epoch,average_time_to_predict]')
        print ([dirname,batch_size,average_time_per_epoch,average_time_to_predict])
        print ('--------------------------------------------------------------------')
        
        # Close the Session when we're done.
        sess.close()
        
            

# Compare models' accuracy, loss and elapsed time per epoch.
plt.ioff()
plt.style.use('ggplot')
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_title('Accuracy')
ax1.set_ylabel('Validation Accuracy')
ax1.set_xlabel('Epochs')
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.set_title('Loss')
ax2.set_ylabel('Validation Loss')
ax2.set_xlabel('Epochs')
ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax3.set_title('Time')
ax3.set_ylabel('Seconds')
for mode, result in zip(batch_sizes, results):
    ax1.plot(result[0].epoch, result[0].history['val_acc'], label=mode)
    ax2.plot(result[0].epoch, result[0].history['val_loss'], label=mode)
ax1.legend()
ax2.legend()
ax3.bar(np.arange(len(results)), [x[1] for x in results],
        tick_label=batch_sizes, align='center')
plt.tight_layout()

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

plt.savefig(out_dir+dirname+'_fig.png')


# with open(out_dir+dirname+"_avg_time_results.csv", "wb") as f:
#      w = csv.writer(f)
#      w.writerows(avg_time_results)
#
# with open(out_dir+dirname+"_results.csv", "wb") as f:
#      w = csv.writer(f)
#      w.writerows(results)
