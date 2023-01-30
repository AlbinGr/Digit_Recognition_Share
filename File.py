import tensorflow as tf
import os
import os.path
from keras.models import load_model
from keras.models import model_from_json


class ModelFile:
    def __init__(self, name="model_file"):
        self.file_name = name
        self.file_path= 'Files/' + self.file_name + '.h5'

    def save(self, mod):
        mod.save(self.file_path)
        print("Saved model to disk")

    def load(self):
        """  The load function needs to recompile the model.
          For now default parameters are entered but this might need
          to be changed  """

        loaded_model = tf.keras.models.load_model(self.file_path)
        print("Loaded model from disk: " + self.file_name)
        return loaded_model

    def del_model(self):
        os.remove(self.file_path)
        print('Files deleted: ' + self.file_name)

    def file_exists(self):
        return os.path.isfile(self.file_path)


