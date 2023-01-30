import tensorflow as tf
import time
import copy
from keras.layers import Dense, Flatten
from keras import Model, callbacks

from matplotlib import pyplot as plt


class Test():

    def __init__(self,
                 mod,
                 loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'],
                 optimizer='adam'):
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = mod

    def accuracy_plot(self, X, y, epochs=1, rate=0.2, X_valid=None, y_valid=None, callback=False):
        mod = tf.keras.models.clone_model(self.model)
        mod.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

        if X_valid is None or y_valid is None:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs, callbacks=[call])
            else:
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs)
        else:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[call])
            else:
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid))

        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return hist.history

    def loss_plot(self, X, y, epochs=1, rate=0.2, X_valid=None, y_valid=None, callback=False):
        mod = tf.keras.models.clone_model(self.model)
        mod.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
        if X_valid is None or y_valid is None:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs, callbacks=[call])
            else:
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs)
        else:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[call])
            else:
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid))

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return hist.history

    def epochs_time(self, X, y, epochs=4, verbose=False):
        mod = tf.keras.models.clone_model(self.model)
        mod.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

        start = time.time()
        mod.fit(X, y, epochs=epochs)
        end = time.time()
        if verbose:
            print('Average epoch tims: ' + str((end-start)/epochs) + 'ms')
        return (end-start)/epochs

    def total_time(self, X, y, epochs=10, verbose=False):
        mod = tf.keras.models.clone_model(self.model)
        mod.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

        call = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        start = time.time()
        mod.fit(X, y, epochs=epochs, callbacks=[call])
        end = time.time()
        if verbose:
            print('Training time: ' + str((end - start)) + 'ms')
        return end - start

    def result(self, X, y, epochs=1, rate=0.2, X_valid=None, y_valid=None, callback=False):
        mod = tf.keras.models.clone_model(self.model)
        mod.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
        t = 0
        if X_valid is None or y_valid is None:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
                t = time.time()
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs, callbacks=[call], verbose=0)
                t = time.time() - t
            else:
                t = time.time()
                hist = mod.fit(X, y, validation_split=rate, epochs=epochs, verbose=0)
                t = time.time() - t
        else:
            if callback:
                call = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
                t = time.time()
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[call], verbose=0)
                t = time.time() - t
            else:
                t = time.time()
                hist = mod.fit(X, y, epochs=epochs, validation_data=(X_valid, y_valid), verbose=0)
                t = time.time()-t

        return hist.history, t


