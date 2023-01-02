import tensorflow as tf

class ImagePreprocessor:
  def __init__(self):
    self.data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input