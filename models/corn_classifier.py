import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from base.base_model import BaseModel
from preprocessing.preprocessor import ImagePreprocessor


class CornDiseaseClassifier(BaseModel):
    def __init__(self, config):
        super(CornDiseaseClassifier, self).__init__(config)
        self.build_model()
        self.compile(tf.keras.optimizers.deserialize(self.config.model.optimizer), 
                     self.config.model.losses, ['accuracy'])

    def build_model(self):
        self.preprocessor = ImagePreprocessor()
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        self.base_model = MobileNetV2(input_shape=(self.config.data_loader.input_size, 
                                                   self.config.data_loader.input_size, 3),
                                      include_top=False,
                                      weights='imagenet'
                                    )
        self.base_model.trainable = False
        
        self.inputs = tf.keras.Input(shape=(self.config.data_loader.input_size, 
                                            self.config.data_loader.input_size, 3))
        
        self.x = self.preprocessor.data_augmentation(self.inputs)
        self.x = self.preprocessor.preprocess_input(self.x)
        self.x = self.base_model(self.x, training=False)

        self.x = self.global_average_layer(self.x)

        self.x = tf.keras.layers.Dense(1024, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)

        self.x = tf.keras.layers.Dense(512, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)

        self.x = tf.keras.layers.Dense(256, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)                  

        self.x = tf.keras.layers.Dense(2, activation='softmax')(self.x) 

        self.model = tf.keras.Model(self.inputs, self.x)

        # self.last_output = self.base_model.output

        # self.x = tf.keras.layers.Flatten(name='flatten')(self.last_output)
        # self.x = tf.keras.layers.Dense(32, activation='relu')(self.x)
        # self.x = tf.keras.layers.Dropout(0.2)(self.x)
        # self.x = tf.keras.layers.Dense(64, activation='relu')(self.x)
        # self.x = tf.keras.layers.Dropout(0.2)(self.x)
        # self.x = tf.keras.layers.Dense(128, activation='relu')(self.x)
        # self.x = tf.keras.layers.Dropout(0.5)(self.x)
        # self.x = tf.keras.layers.Dense(2, activation='softmax')(self.x)

        # self.model = tf.keras.models.Model(self.base_model.input, self.x)

        return self.model
    
    def compile(self, optimizer, loss, metrics):
        self.build_model().compile(optimizer=optimizer, loss=loss, metrics=metrics)