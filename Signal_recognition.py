import os
import time
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint

import keras.backend as K
import random
from scipy.io import loadmat
from scipy.io import savemat
from SelfDesign import plot_confusion_matrix

import h5py

import warnings
warnings.filterwarnings('ignore')

# K.clear_session()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1 for CPU；  0 for GPU

start_time = time.time()
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
layer = 6

# Define the traning parameters
lam = 1550e-9
pixel = 8e-6
d_detector = 150e-3
d_layer = 100e-3
print(d_detector)
print(d_layer)

nu1 = 1  
d_num = 100 
zl_num = 1 
diff_num = 1

w0 = 5

keep_training = 1# False :0  True:1
retrain = 0
before_output = 0

construct = 1
lenlayer = int(np.sqrt(layer))
output_dim = 2

size_big = 200
epochs = 200
factor = 99

checkpoint_path = './pointcheck/signal2_layer'+str(layer)+'_zl'+str(zl_num)+'_din'+str(d_num)+'_OAM'+str(factor)

file_path1 = './sig/new50GB_signal_no1_zl1_din100_l1.mat'#ook
file_path2 = './sig/new50GB_signal_no2_zl1_din100_l1.mat'#cap

pict_name = './sig/signal_OUT'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat' #all Out pict
num_name = './sig/num_OUT'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat' #all Out num
weights_name = './sig/signal_weight'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat'    # weight out

file_name = './sig/pam4/pam4signal_out'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat'
pict0_name = './sig/pam4/pam4signal0_after'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat'
pict1_name = './sig/pam4/pam4signal1_after'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat'
weights0_name = './sig/pam4/pam4signal_weight'+str(factor)+'_zl'+str(zl_num)+'_w0'+str(w0)+'.mat'

save_path = "./loss_record_recog.mat"

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
print("shape of data_PRE")

def normalize(file_path):
    print(file_path)
    data_Rx_prep = dataprocessing(file_path)
    print(np.shape(data_Rx_prep))

    print(np.max(data_Rx_prep))
    print(np.min(data_Rx_prep))
    return data_Rx_prep

if output_dim == 2:
    data_Rx_prep = normalize(file_path1)
    data_Rx2_prep = normalize(file_path2)

    data_Rx_length = len(data_Rx_prep)
    data_Rx2_length = len(data_Rx2_prep)

    lable_Rx = tf.one_hot(np.ones((data_Rx_length)) * 0, output_dim)
    lable_Rx2 = tf.one_hot(np.ones((data_Rx2_length)) * 1, output_dim)


print("===============Data Loading===============")

ratio = 0.8
if output_dim == 2:
    x_train_temp = np.concatenate(
        [data_Rx_prep[0:int(data_Rx_length * ratio)], data_Rx2_prep[0:int(data_Rx2_length * ratio)]])
    x_test_temp = np.concatenate(
        [data_Rx_prep[int(data_Rx_length * ratio) + 1:], data_Rx2_prep[int(data_Rx2_length * ratio) + 1:]])

    y_train_temp = np.concatenate([lable_Rx[0:int(data_Rx_length * ratio)], lable_Rx2[0:int(data_Rx2_length * ratio)]])
    y_test_temp = np.concatenate([lable_Rx[int(data_Rx_length * ratio) + 1:], lable_Rx2[int(data_Rx2_length * ratio) + 1:]])

waveform_tarin = tf.data.Dataset.from_tensor_slices((x_train_temp, y_train_temp))
waveform_test = tf.data.Dataset.from_tensor_slices((x_test_temp, y_test_temp))

train_dataset = waveform_tarin.map(preprocess).cache().shuffle(len(waveform_tarin)).batch(BATCH_SIZE)
test_dataset = waveform_test.map(preprocess).shuffle(len(waveform_test)).batch(BATCH_SIZE)

print("up")
class SLMmodem(Layer):
    def __init__(self, units=1200):
        super(SLMmodem, self).__init__()
        self.units = units
        self.Nx = units  
        self.dx = pixel
        self.lam = lam  


    def build(self, input_shape):
        phase_init = tf.random_normal_initializer()
        self.phase = tf.Variable(name="phase",
                                 initial_value=phase_init(shape=(self.units, self.units), dtype='float32'),
                                 trainable=True,
                                 constraint=lambda t: tf.clip_by_value(t, 0, 2 * np.pi))

    @tf.function
    def call(self, inputs):
        return tf.multiply(inputs,tf.math.exp(1j * tf.cast(self.phase, dtype=tf.complex64)))  ## PM

class Propogation(Layer):
    def __init__(self, units=1200):
        super(Propogation, self).__init__()
        self.units = units
        self.Nx = units 
        self.dx = pixel
        self.lam = lam  
        self.z = d_detector  

    @tf.function
    def call(self, inputs):
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
    def __init__(self, units=200):
        super(Detector, self).__init__()
        self.units = units

    @tf.function
    def call(self, inputs):
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

        out = (ten_regions(tf.square(tf.square(tf.abs(inputs)))))
        E1 = tf.cast(out, dtype=tf.float32)
        sum_a = tf.reduce_sum(E1, axis=1, keepdims=True)
        E4 = E1 / sum_a

        return E4

def complex_relu(z):
    return tf.complex(tf.nn.relu(tf.math.real(z)),
                     tf.nn.relu(tf.math.imag(z)))

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
                              patience=5, min_lr=0.001, mode='auto')

EarlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.00000000001,
                          patience=epochs * 0.5 * 2, verbose=1, mode='auto')

if retrain:
    history = D2NN.fit(train_dataset,
                       epochs=epochs,
                       validation_data=test_dataset,
                       callbacks=[checkpoint, reduce_lr, EarlyStop],
                       use_multiprocessing=True,
                       workers=8,
                       max_queue_size=20)

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    epoch_axis = np.arange(1, len(train_loss) + 1)

    savemat(save_path, {
        'epoch': epoch_axis,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    })
   
print("===============Going to predict===============")
if output_dim == 2:
    x_predict = x_train_temp
    y_predict = x_test_temp 

waveform_predict = tf.data.Dataset.from_tensor_slices((x_predict, y_predict))
D2NN.load_weights(checkpoint_path)
predict_Data_final = waveform_predict.map(preprocess).batch(BATCH_SIZE)
data_class= D2NN.predict(predict_Data_final)

numall = {'my_array': data_class}
savemat(num_name, numall)
savemat(file_name, numall)

data_class_ = np.argmax(data_class, axis=1)
data_ture_ = np.argmax(y_predict, axis=1)

# Plot confusion_matrix
fig3 = plt.figure(figsize=(7, 7))
conf_mat = tf.math.confusion_matrix(data_ture_, data_class_, output_dim)
plot_confusion_matrix(np.array(conf_mat), normalize=False, target_names=['OOK', 'CAP8'],
                      title='Confusion Matrix')

layer_output = D2NN.layers[2].output  
intermediate_layer_model = tf.keras.Model(inputs=D2NN.input, outputs=layer_output)
recog_PICT = intermediate_layer_model.predict(predict_Data_final)
print(np.shape(recog_PICT))
data0 = {'my_array': recog_PICT}
savemat(pict_name, data0)

data2 = {'my_array': recog_PICT[1]}
savemat(pict0_name, data2)

data3 = {'my_array': recog_PICT[size_big+10]}   
savemat(pict1_name, data3)

data_output_abs0 = (abs(recog_PICT[1])) ** 1
fig4 = plt.figure(figsize=(7, 7))
plt.imshow(data_output_abs0,cmap='RdBu')
plt.colorbar()

data_output_abs = (abs(recog_PICT[size_big+10])) ** 1
fig5 = plt.figure(figsize=(7, 7))
plt.imshow(data_output_abs,cmap='RdBu')
plt.colorbar()


print("===============Plot the region of the ten detectors===============")
detector_plane = np.zeros([size, size])
if output_dim == 2:
    ten_detector_regions = [[190, 200, 245, 255],
                            [300, 310, 245, 255]]


def get_range(shape, size=size, base=500):
    x0 = shape[0] * size // base
    y0 = shape[2] * size // base
    delta = (shape[1] - shape[0]) * size // base
    x1 = x0 + delta
    y1 = y0 + delta
    return x0, x1, y0, y1
detector_point = list()
for i in range(0, output_dim):
    x0, x1, y0, y1 = get_range(ten_detector_regions[i])
    # print([x0,x1,y0,y1])
    detector_plane[x0:x1, y0:y1] = 1
    detector_point.append(
        [[x0 - 0.5, x1 - 0.5, x1 - 0.5, x0 - 0.5, x0 - 0.5], [y0 - 0.5, y0 - 0.5, y1 - 0.5, y1 - 0.5, y0 - 0.5]])
# plot the detector region
rows, cols = 1, 1
fig7 = plt.figure(figsize=(7, 7))
plt.imshow(detector_plane, cmap='gray')
plt.show()

plt.figure(fig7)
for i in range(0, output_dim):
    plt.plot(detector_point[i][1], detector_point[i][0], 'b-')
plt.colorbar()
plt.axis('off')
plt.show()
# calculate the detector
def get_range_arr(arr, shape, size=size, base=500):
    x0 = shape[0] * size // base
    y0 = shape[2] * size // base
    delta = (shape[1] - shape[0]) * size // base
    return arr[x0:x0 + delta, y0:y0 + delta]

#
detector_point_value = list()
for i in range(0, output_dim):
    temp = get_range_arr(recog_PICT[8], ten_detector_regions[i])
    detector_point_value.append(np.mean(abs(temp)))
print("===============All Done===============")

end_time = time.time()
total_time = end_time - start_time
if total_time < 3600:
    print("running time is {:.2f} minutes".format(total_time / 60))
else:
    print("running time is {:.2f} hours".format(total_time / 3600))

if retrain == 1:
    plt.figure()
    np.savetxt('epoch.txt', history.epoch, fmt='%1.7e')
    np.savetxt('train_accuracy.txt', history.history['accuracy'], fmt='%1.7e')
    np.savetxt('val_accuracy.txt', history.history['val_accuracy'], fmt='%1.7e')
    plt.plot(history.epoch, history.history['accuracy'], 'r')
    plt.plot(history.epoch, history.history['val_accuracy'], 'b')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train accuracy ', 'Validation accuracy'])
    plt.show()

print("===============Geting the training out===============")

# dataall = {'my_array': equalized_PICT}
# savemat(pict_name, dataall)

if construct:

    weights = []
    weight1 = D2NN.layers[1].get_weights()
    weight1 = np.array(weight1)
    weights = weight1

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