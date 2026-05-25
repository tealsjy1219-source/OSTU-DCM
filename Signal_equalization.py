import os
import time
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.special import genlaguerre
import math

from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint

import keras.backend as K
import random
from scipy.io import loadmat
from scipy.io import savemat

import h5py

import warnings
warnings.filterwarnings('ignore')

K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1 for CPU；  0 for GPU

start_time = time.time()
SEED = 27
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)

save_path = "./loss_record_equal.mat"

size = 130
BUFFER_SIZE = 256
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
learning_rate = 0.01
layer = 4
construct = 1
# Define the traning parameters
lam = 1550e-9
pixel = 1e-6
d_detector = 100e-6
d_layer = 50e-6

v_num = 50
zl_num = 1
d_num = 50
pre_num = 0
data_num = 584
opt_num = 60 

# load the dataset
keep_training = 1 # False:0  True:1
retrain = 0
before_output = 0

size_big = 5000
epochs = 100

checkpoint_path = './pointcheck/D2NN_'+str(data_num)+'_up128_opt'+str(opt_num)+'_OAM'

name_Tx = ['./sig/pam4/PAM4_26G', '.txt']#30
name_Tx = ''.join(name_Tx)
data_Tx_prep = np.loadtxt(name_Tx)
data_Tx_prep = data_Tx_prep

file_path = './sig/pam4/pam4_26G_um_zl1_din50_l1.mat'

file_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_out'+str(opt_num)+'.mat'
pict_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_after'+str(opt_num)+'.mat'
weights_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_weight'+str(opt_num)+'.mat'
train_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_train'+str(opt_num)+'.mat'

Lenout_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_lenout'+str(opt_num)+'.mat'
lenpict_name = './sig/pam4/pam4_'+str(data_num)+'_n'+str(pre_num)+'_lenafter'+str(opt_num)+'.mat'
print("===============Data Processing===============")
def preprocess(image, label):
    image_t = tf.cast(image, tf.complex64)
    return tf.cast(image_t, dtype=tf.complex64), label
def dataprocessing(file_path):
    with h5py.File(file_path, 'r') as f:
        data = np.array(f['yt'])
        if data.dtype.names == ('real', 'imag'): 
            real_part = data['real']
            imag_part = data['imag']
            data_Rx_prep = real_part + 1j * imag_part
    data_Rx_prep = data_Rx_prep.transpose(2, 0, 1)
    data_Rx_prep = data_Rx_prep[0:size_big, :, :]
    print(np.shape(data_Rx_prep))
    return data_Rx_prep

print(np.shape(data_Tx_prep))
print(file_path)
data_Rx_prep = dataprocessing(file_path)
length = len(data_Rx_prep)

start = 5
data_Tx_prep = data_Tx_prep[start:length+start]

data_Tx_prep = (data_Tx_prep*1+5)/2
datamax = np.max(np.abs(data_Tx_prep))
data_Tx_prep = data_Tx_prep/datamax


def normalize(data_Rx_prep,data_Tx_prep):
    print(np.shape(data_Tx_prep))
    datamint = np.min(data_Rx_prep)
    data_Rx_prep = data_Rx_prep - datamint
    datamaxt = np.max(np.abs(data_Rx_prep))
    data_Rx_prep = data_Rx_prep / datamaxt

    print("shape of data_PRE")
    print(np.max(data_Tx_prep))
    print(np.min(data_Tx_prep))

    print("shape of data_AFTER")
    print(np.max(data_Rx_prep))
    print(np.min(data_Rx_prep))
    return data_Rx_prep,data_Tx_prep

data_Rx_prep,data_Tx_prep = normalize(data_Rx_prep,data_Tx_prep)

print("===============Data Loading===============")

data_Rx_length = len(data_Rx_prep)
ratio = 0.8


x_train_temp2 = data_Rx_prep[0:int(data_Rx_length * ratio)]
y_train_temp2 = data_Tx_prep[0:int(data_Rx_length * ratio)]
x_test_temp3 = data_Rx_prep[int(data_Rx_length * ratio) + 1:]
y_test_temp3 = data_Tx_prep[int(data_Rx_length * ratio) + 1:]

waveform_train2 = tf.data.Dataset.from_tensor_slices((x_train_temp2, y_train_temp2))
waveform_test3 = tf.data.Dataset.from_tensor_slices((x_test_temp3, y_test_temp3))
train_dataset2 = waveform_train2.map(preprocess).cache().shuffle(len(waveform_train2)).batch(BATCH_SIZE)
test_dataset3 = waveform_test3.map(preprocess).batch(BATCH_SIZE)

print("up")


class SLMmodem(Layer):
    def __init__(self, units=1200):
        '''Initialize the diffraction layer attributes'''
        super(SLMmodem, self).__init__()
        self.units = units
        self.Nx = units             
        self.dx = pixel
        self.lam = lam  

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        phase_init = tf.random_normal_initializer()
        self.phase = tf.Variable(name="phase",
                                 initial_value=phase_init(shape=(self.units, self.units), dtype='float32'),
                                 trainable=True,
                                 constraint=lambda t: tf.clip_by_value(t, 0, 2 * np.pi))

    @tf.function
    def call(self, inputs):
        phase= self.phase
        return tf.multiply(inputs,tf.math.exp(1j * tf.cast(phase, dtype=tf.complex64)))


class Propogation(Layer):
    def __init__(self, units=1200):
        '''Initialize the diffraction layer attributes'''
        super(Propogation, self).__init__()
        self.units = units
        self.Nx = units  
        self.dx = pixel
        self.lam = lam  
        self.z = d_detector 

    @tf.function
    def call(self, inputs):
        '''Define the computation'''

        def angular_spectrum_propagator(E, z=self.z, lam=self.lam):
            fft_c = tf.signal.fft2d(E)
            c = tf.signal.fftshift(fft_c)

            fx = np.fft.fftshift(np.fft.fftfreq(self.Nx, d=self.dx))
            fxx, fyy = np.meshgrid(fx, fx)
            argument = (2 * np.pi) ** 2 * ((1. / lam) ** 2 - fxx ** 2 - fyy ** 2)

            tmp = np.sqrt(np.abs(argument))
            kz = np.where(argument >= 0, tmp, 1j * tmp)

            E = tf.signal.ifft2d(tf.signal.ifftshift(c * np.exp(1j * kz * z)))  # phase是加还是减
            return E

        return angular_spectrum_propagator(inputs)


class Detector(Layer):
    def __init__(self, units=1200):
        '''Initialize the instance attributes'''
        super(Detector, self).__init__()
        self.units = units

    @tf.function
    def call(self, inputs):
        ''' Converts output to one hot form
        Applies softmax'''

        def rang(arr, shape, size=size, base=800):
            x0 = shape[0] * size // base
            y0 = shape[2] * size // base
            delta = (shape[1] - shape[0]) * size // base
            return arr[x0:x0 + delta, y0:y0 + delta]

        def reduce_mean(tf_):
            return tf.reduce_mean(tf_)

        def _ten_regions(a):
            return tf.map_fn(reduce_mean, tf.convert_to_tensor([
                rang(a, (390, 410, 390, 410)),
            ]))

        def ten_regions(logits):
            return tf.map_fn(_ten_regions, tf.abs(logits), dtype=tf.float32)
        Eout = ten_regions(tf.square(tf.abs(inputs)))
        return Eout 

def get_D2NN_model():
    inputs = keras.layers.Input(shape=[size, size], dtype='complex64')
    h1 = SLMmodem(size)(inputs)
    propogation = Propogation(size)(h1)
    out = Detector()(propogation)
    return tf.keras.Model(inputs, out)

D2NN = get_D2NN_model()
D2NN.summary()
if keep_training:
    D2NN.load_weights(checkpoint_path)

def loss_function(y_ture,y_pred):

    y_true_squared_sum = tf.reduce_sum(tf.square(y_ture))
    y_pred_squared_sum = tf.reduce_sum(tf.square(y_pred))

    scale_factor = tf.reduce_sum(tf.multiply(y_ture, y_pred)) / (y_pred_squared_sum)
    nmse = tf.reduce_mean(tf.square(y_ture - scale_factor * y_pred))

    Pout = tf.reduce_sum(tf.square(y_pred))
    P1 = tf.reduce_sum(tf.square(y_ture))
    eta1 = Pout / (P1 * 10000)
    Lp = tf.exp(-1 * eta1)

    out1 = 2*nmse+10*Lp
    return out1

D2NN.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
             loss =loss_function,
             # loss = 'mean_squared_error',
             metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             monitor='val_loss',  
                             save_best_only=True,  
                             mode='min',  
                             verbose=1)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90, verbose=1,
                              patience=5, min_lr=0.000001, mode='auto')

EarlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.00000000001,
                          patience=epochs * 0.5 * 2, verbose=1, mode='auto')
if retrain:
    history = D2NN.fit(train_dataset2,
                       epochs=epochs,
                       validation_data=test_dataset3,
                       callbacks=[checkpoint, reduce_lr, EarlyStop],
                       use_multiprocessing=True,
                       workers=8,
                       max_queue_size=20)

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    epoch_axis = np.arange(1, len(train_loss) + 1)

    savemat(save_path, {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch_axis
    })
 

print("===============Going to predict===============")

D2NN.load_weights(checkpoint_path)
predict_Data_final = test_dataset3
equalized_signal = D2NN.predict(predict_Data_final)
print(np.shape(equalized_signal))

data = {'my_array': equalized_signal}
savemat(file_name, data)


layer_output = D2NN.layers[2].output 
intermediate_layer_model = tf.keras.Model(inputs=D2NN.input, outputs=layer_output)
equalized_PICT = intermediate_layer_model.predict(predict_Data_final)
data2 = {'my_array': equalized_PICT[0:2]}
print(np.shape(equalized_PICT))
savemat(pict_name, data2)


if construct:

    weights = []
    weight1 = D2NN.layers[1].get_weights()
    weight1 = np.array(weight1)
    print(np.shape(weight1))

    weights = weight1
    print(np.shape(weights))

    # Map the weights into range [0, 2*pi]
    for i in range(0, size):
        for j in range(0, size):
            while (weights[0][i][j] < 0):
                weights[0][i][j] += 2 * np.pi
            while (weights[0][i][j] > 2 * np.pi):
                weights[0][i][j] -= 2 * np.pi
    print("The max value in wights is: " + str(np.max(weights)))
    print("The mim value in wights is: " + str(np.min(weights)))
    data = {'weight': weights}
    savemat(weights_name, data)
