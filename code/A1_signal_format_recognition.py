import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from scipy.io import savemat
from SelfDesign import plot_confusion_matrix,dataprocessing,preprocess

import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

SEED = 27
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)

after_size = 0
size = 1020
BUFFER_SIZE = 28
BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
learning_rate = 0.01

lam = 1550e-9
pixel = 8e-6
d_detector = 150e-3
d_layer = 100e-3

nu1 = 1  
d_num = 100 
zl_num = 1 

keep_training = 1
retrain = 0

construct = 1
output_dim = 2

size_big = 500
epochs = 200
factor = 101
print(factor)

checkpoint_path = './pointcheck/signal2_l'+str(nu1)+'_zl'+str(zl_num)+'_din'+str(d_num)+'_factor'+str(factor)

file_path1 = './data/signal_l'+str(nu1)+'_no1_OOK_zl'+str(zl_num)+'_din'+str(d_num)+'.mat'    #ook
file_path2 = './data/signal_l'+str(nu1)+'_no2_CAP16_zl'+str(zl_num)+'_din'+str(d_num)+'.mat'  #cap

weights_name = './output/signal2_weight'+str(factor)+'_zl'+str(zl_num)+'_l'+str(nu1)+'.mat'   

print("===============Data Processing===============")

def normalize(file_path,size_big):
    print(file_path)
    data_Rx_prep = dataprocessing(file_path,size_big)
    print(np.shape(data_Rx_prep))

    print(np.max(data_Rx_prep))
    print(np.min(data_Rx_prep))
    return data_Rx_prep

print("shape of data_PRE")
data_Rx_prep = normalize(file_path1,size_big)
data_Rx2_prep = normalize(file_path2,size_big)

data_Rx_length = len(data_Rx_prep)
data_Rx2_length = len(data_Rx2_prep)

lable_Rx = tf.one_hot(np.ones((data_Rx_length)) * 0, output_dim)
lable_Rx2 = tf.one_hot(np.ones((data_Rx2_length)) * 1, output_dim)

print("===============Data Loading===============")

ratio = 0.6
split_idx = int(data_Rx_length * ratio)

x_train_temp = np.concatenate(
    [data_Rx_prep[0:split_idx], data_Rx2_prep[0:split_idx]])
x_test_temp = np.concatenate(
    [data_Rx_prep[split_idx:], data_Rx2_prep[split_idx:]])

y_train_temp = np.concatenate([lable_Rx[0:split_idx], lable_Rx2[0:split_idx]])
y_test_temp = np.concatenate([lable_Rx[split_idx:], lable_Rx2[split_idx:]])

waveform_tarin = tf.data.Dataset.from_tensor_slices((x_train_temp, y_train_temp))
waveform_test = tf.data.Dataset.from_tensor_slices((x_test_temp, y_test_temp))

train_dataset = waveform_tarin.map(preprocess).cache().shuffle(len(waveform_tarin)).batch(BATCH_SIZE)
test_dataset = waveform_test.map(preprocess).shuffle(len(waveform_test)).batch(BATCH_SIZE)

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
        '''Define the computation'''
        return tf.multiply(inputs,tf.math.exp(1j * tf.cast(self.phase, dtype=tf.complex64)))  ## PM

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
    def __init__(self, units=200):
        '''Initialize the instance attributes'''
        super(Detector, self).__init__()
        self.units = units

    @tf.function
    def call(self, inputs):
        ''' Converts output to one hot form
        Applies softmax'''

        def rang(arr, shape,base=500):
            x0 = shape[0] * size // base
            y0 = shape[2] * size // base
            delta = (shape[1] - shape[0]) * size // base
            return arr[x0:x0 + delta, y0:y0 + delta]

        def reduce_mean(tf_):
            return tf.reduce_mean(tf_)

        def _ten_regions(a):
                return tf.map_fn(reduce_mean, tf.convert_to_tensor([
                    rang(a, (190, 200, 245, 255)),
                    rang(a, (300, 310, 245, 255)),
                ]))

        def ten_regions(logits):
            return tf.map_fn(_ten_regions, tf.abs(logits), dtype=tf.float32)

        out = (ten_regions(tf.square((tf.abs(inputs)))))
        E1 = tf.cast(out, dtype=tf.float32)
        sum_a = tf.reduce_sum(E1, axis=1, keepdims=True)
        E4 = E1 / sum_a
        return E4

def get_D2NN_model():
    inputs = keras.layers.Input(shape=[size, size], dtype='complex64')
    h1 = SLMmodem(size)(inputs)
    propogation = Propogation(size)(h1)
    out = Detector(size)(propogation)
    return tf.keras.Model(inputs, out)

D2NN = get_D2NN_model()   
D2NN.summary()

if keep_training:
    D2NN.load_weights(checkpoint_path)

D2NN.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
             loss =tf.keras.losses.BinaryCrossentropy(from_logits=False),
             # loss='mse',
             metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max',  
                             verbose=1)  
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.90, verbose=1,
                              patience=5, min_lr=0.000001, mode='auto')

EarlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.00001,
                          patience=epochs * 0.5, verbose=1, mode='auto')

if retrain:
    history = D2NN.fit(train_dataset,
                       epochs=epochs,
                       validation_data=test_dataset,
                       callbacks=[checkpoint, reduce_lr, EarlyStop],
                       use_multiprocessing=True,
                       workers=8,
                       max_queue_size=20)

print("===============Going to predict===============")
x_train_predict =  x_test_temp
y_train_predict =  y_test_temp

waveform_predict = tf.data.Dataset.from_tensor_slices((x_train_predict, y_train_predict))
D2NN.load_weights(checkpoint_path)
predict_Data_final = waveform_predict.map(preprocess).batch(BATCH_SIZE)
data_class= D2NN.predict(predict_Data_final)
print(np.shape(data_class))

data_class_ = np.argmax(data_class, axis=1)
data_ture_ = np.argmax(y_train_predict, axis=1)

# Plot confusion_matrix
conf_mat = tf.math.confusion_matrix(data_ture_, data_class_, output_dim)
plot_confusion_matrix(np.array(conf_mat), normalize=False, target_names=['OOK', 'CAP16'],
                      title='Confusion Matrix')


layer_output = D2NN.layers[2].output  
intermediate_layer_model = tf.keras.Model(inputs=D2NN.input, outputs=layer_output)
equalized_PICT = intermediate_layer_model.predict(predict_Data_final)
print(np.shape(equalized_PICT))

data_output_abs = (abs(equalized_PICT[1]))
fig5= plt.figure(figsize=(7,7))
plt.imshow(data_output_abs,cmap='RdBu')
plt.colorbar()


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

    fig33 = plt.figure(figsize=(7, 7))
    plt.imshow(weights[0], cmap='RdBu')
    data = {'weight': weights}
    savemat(weights_name, data)

    weight_output_abs = (abs(weights[0]))
    fig6= plt.figure(figsize=(7,7))
    plt.imshow(weight_output_abs,cmap='RdBu')
    plt.colorbar()
    plt.show()

    print("===============All Done===============")