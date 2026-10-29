import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

def preprocess(image, label):
    image_t = tf.cast(image, tf.complex64)
    return tf.cast(image_t, dtype=tf.complex64), label

def dataprocessing(file_path,size_big):
    with h5py.File(file_path, 'r') as f:
        data = np.array(f['yt'])
        if data.dtype.names == ('real', 'imag'): 
            real_part = data['real']
            imag_part = data['imag']
            data_Rx_prep = real_part + 1j * imag_part
    data_Rx_prep = data_Rx_prep.transpose(2, 0, 1)
    data_Rx_prep = data_Rx_prep[0:0+size_big, :, :]
    return data_Rx_prep

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,  #  Greens Reds viridis Oranges
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(2,figsize=(6, 5))
    # plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
   
    # plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.tight_layout()
    plt.show()
