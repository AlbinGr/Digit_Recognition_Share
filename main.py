import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import sys

from PyQt5.QtWidgets import QApplication
from interface import Window

from keras.datasets import mnist
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from Testing import Test
from File import ModelFile

"""network creation"""


def network_creation(layers=[128],
                     activation='relu',
                     optimizer='adam',
                     drop=0.2,
                     loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy']
                     ):
    # Define the model

    mod = tf.keras.Sequential()
    mod.add(Conv2D(64, (3, 3), activation=activation, input_shape=(28, 28, 1)))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Conv2D(32, (3, 3), activation=activation))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Flatten())
    for n in layers:
        mod.add(tf.keras.layers.Dense(n, activation=activation))
        mod.add(tf.keras.layers.Dropout(drop))

    mod.add(Dense(10, activation=activation))

    # Loss function
    # The SparseCategoricalCrossEntropy is adapted to this particular problem
    mod.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return mod


"""network training"""


def network_training(mod, x, y, epochs, mod_file, X_valid, y_valid):
    call = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    hist = mod.fit(x, y, epochs=epochs, callbacks=[call], validation_data=(X_valid, y_valid))
    mod_file.save(mod)
    return hist


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    """saving and uploading the model from ModelFile object"""
    myname = "model_file"
    file = ModelFile(myname)
    if not file.file_exists():
        model = network_creation()
        history = network_training(model, x_train, y_train, 10, file, x_test, y_test)
    else:
        model = file.load()

    """Call the application"""
    app = QApplication(sys.argv)
    form = Window(model)
    app.exec_()

""" This code was used to test and display the validation and training accuracy of different models 
 """


def test_layer(test_list=[[128]]):
    """ Test of changing number of layers and neurons in each layer """
    # test_list = [[64], [128], [256], [64, 64], [128, 128]]

    epochs = 12
    dropout = 0.2
    callback = True
    time = {str(t): 0 for t in test_list}
    hist = {str(t): None for t in test_list}
    for nt in test_list:
        print("Testing network with " + str(len(test_list[test_list.index(nt)])) +
              " hidden layers with " + str(test_list[test_list.index(nt)][0]) + ' neurons. ')
        test_obj = Test(network_creation(layers=nt))
        hist[str(nt)], time[str(nt)] = test_obj.result(x_train,
                                                       y_train,
                                                       epochs=epochs,
                                                       X_valid=x_test,
                                                       y_valid=y_test,
                                                       callback=True)
        print('Fitting time: ' + str(time[str(nt)]) + ' s')

    index = [str(t) for t in test_list]
    accuracy = pd.DataFrame({'train accuracy': [hist[str(t)]['accuracy'][len(hist[str(t)]['accuracy']) - 1]
                                                for t in test_list],
                             'validation accuracy': [hist[str(t)]['val_accuracy']
                                                     [len(hist[str(t)]['val_accuracy']) - 1] for t in test_list]},
                            index=index)

    X_axis = np.arange(len(index))
    X_label = []
    for i in range(len(test_list)):
        txt = str(len(test_list[i])) + "x" + str(test_list[i][0])
        X_label.append(txt)
    plt.subplot(121)
    plt.bar(X_axis - 0.2, accuracy.loc[:, 'train accuracy'], 0.4, label='Train accuracy')
    plt.bar(X_axis + 0.2, accuracy.loc[:, 'validation accuracy'], 0.4, label='Validation accuracy')

    plt.xticks(X_axis, X_label)
    plt.xlabel("Network type")
    plt.ylabel("Accuracy")
    plt.title("Validation and Train accuracy")
    plt.legend()

    plt.subplot(122)
    plt.bar(X_axis, time.values())
    plt.xticks(X_axis, X_label)
    plt.xlabel("Network type")
    plt.ylabel("Fitting time (s)")
    plt.title("Evaluation of different networks")
    plt.suptitle('Training time')
    plt.show()
