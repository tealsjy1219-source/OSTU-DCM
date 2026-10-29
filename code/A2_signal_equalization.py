import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.layers import Layer,Lambda,InputLayer, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy.io import savemat
from SelfDesign import dataprocessing,preprocess

import warnings
warnings.filterwarnings('ignore')
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1 for CPU；  0 for GPU

SEED = 27
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)

size = 130
BUFFER_SIZE = 256
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
learning_rate = 0.01

lam = 1550e-9
pixel = 1e-6
d_detector = 50e-6
d_layer = 50e-6

v_num = 50
zl_num = 1
d_num = 50
pre_num = 0
data_num = 588
opt_num = 411

construct = 1
nu1 = 1

keep_training = 1 
retrain = 0

size_big = 2000
epochs = 100
print(opt_num)

checkpoint_path = './pointcheck/pam4_equ_'+str(data_num)+'_zl'+str(zl_num)+'_din'+str(d_num)+'_l'+str(nu1)+'_opt'+str(opt_num)

name_Tx = ['./data/PAM4_26G', '.txt']
name_Tx = ''.join(name_Tx)
data_Tx_prep = np.loadtxt(name_Tx)
data_Tx_prep = data_Tx_prep

file_path = './data/pam4_'+str(data_num)+'_zl'+str(zl_num)+'_din'+str(d_num)+'_l'+str(nu1)+'.mat'
file_name = './output/pam4_'+str(data_num)+'_n'+str(pre_num)+'_out'+str(opt_num)+'.mat'
weights_name = './output/pam4_'+str(data_num)+'_n'+str(pre_num)+'_weight'+str(opt_num)+'.mat'

print("===============Data Processing===============")

print("shape of data_PRE")
print(np.shape(data_Tx_prep))
print(file_path)

data_Rx_prep = dataprocessing(file_path,size_big)
length = len(data_Rx_prep)

start = 4
data_Tx_prep = data_Tx_prep[start:length+start]
print(np.shape(data_Tx_prep))

data_Tx_prep = (data_Tx_prep*1+5)/2

datamax = np.max(np.abs(data_Tx_prep))
data_Tx_prep = data_Tx_prep/datamax
data_Tx_prep = data_Tx_prep[0:length]
print(np.shape(data_Tx_prep))

def normalize(data_Rx_prep,data_Tx_prep):
    print(np.shape(data_Tx_prep))
    datamint = np.min(data_Rx_prep)
    data_Rx_prep = data_Rx_prep - datamint
    datamaxt = np.max(np.abs(data_Rx_prep))
    data_Rx_prep = data_Rx_prep / datamaxt

    print(np.max(data_Tx_prep))
    print(np.min(data_Tx_prep))
    print(np.max(data_Rx_prep))
    print(np.min(data_Rx_prep))
    return data_Rx_prep,data_Tx_prep
print("shape of data_AFTER")
data_Rx_prep,data_Tx_prep = normalize(data_Rx_prep,data_Tx_prep)

print("===============Data Loading===============")

data_Rx_length = len(data_Rx_prep)
ratio = 0.8

split_idx = int(data_Rx_length * ratio)

x_train_temp2 = data_Rx_prep[:split_idx]
y_train_temp2 = data_Tx_prep[:split_idx]

x_test_temp3 = data_Rx_prep[split_idx:]
y_test_temp3 = data_Tx_prep[split_idx:]

if retrain == 1:
    waveform_train2 = tf.data.Dataset.from_tensor_slices((x_train_temp2, y_train_temp2))
    waveform_test3 = tf.data.Dataset.from_tensor_slices((x_test_temp3, y_test_temp3))
    train_dataset2 = waveform_train2.map(preprocess).cache().shuffle(len(waveform_train2)).batch(BATCH_SIZE)
    test_dataset3 = waveform_test3.map(preprocess).batch(BATCH_SIZE)

if retrain == 0:
    waveform_all = tf.data.Dataset.from_tensor_slices((x_test_temp3, y_test_temp3))
print("===============Seting up===============")

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
        '''Define the computation'''
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

            E = tf.signal.ifft2d(tf.signal.ifftshift(c * np.exp(1j * kz * z)))  
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

    epsilon = tf.keras.backend.epsilon()
    scale_factor = tf.reduce_sum(tf.multiply(y_ture, y_pred)) / (y_pred_squared_sum + epsilon)
    af = tf.reduce_mean(tf.square(y_ture - scale_factor * y_pred))

    Pout = tf.reduce_sum(tf.square(y_pred))
    P1 = tf.reduce_sum(tf.square(y_ture))
    eta1 = Pout / (P1 * 5000)
    Lp2 = tf.exp(-1 * eta1)

    out1 = 2*af+10*Lp2
    return out1

D2NN.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
             loss =loss_function,
            #  loss = 'mean_squared_error',
             metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             monitor='val_loss',  
                             save_best_only=True, 
                             mode='min',  
                             verbose=1)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90, verbose=1,
                              patience=5, min_lr=0.000001, mode='auto')

EarlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.00001,
                          patience=epochs * 0.5, verbose=1, mode='auto')
if retrain:
    history = D2NN.fit(train_dataset2,
                       epochs=epochs,
                       validation_data=test_dataset3,
                       callbacks=[checkpoint, reduce_lr, EarlyStop],
                       use_multiprocessing=True,
                       workers=8,
                       max_queue_size=20)

print("===============Going to predict===============")

D2NN.load_weights(checkpoint_path)
predict_Data_final = waveform_all.map(preprocess).batch(BATCH_SIZE)
equalized_signal = D2NN.predict(predict_Data_final)
print(np.shape(equalized_signal))
data = {'my_array': equalized_signal}
savemat(file_name, data)

if construct:

    weights = []
    weight1 = D2NN.layers[1].get_weights()
    weight1 = np.array(weight1)
    print(np.shape(weight1))

    weights = weight1
    print(np.shape(weights))

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

    print("===============All Done===============")