import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from base.base_model import BaseModel


class CornDiseaseClassifier(BaseModel):
    def __init__(self, config):
        super(CornDiseaseClassifier, self).__init__(config)
        self.build_model()
        self.compile(self.config.model.optimizer, self.config.model.losses, ['accuracy'])

    def build_model(self):
        self.base_model = MobileNetV2(input_shape=(224, 224, 3),
                                      include_top=False,
                                      weights='imagenet'
                                    )
        self.base_model.trainable = False
        self.last_output = self.base_model.output

        self.x = tf.keras.layers.Flatten(name='flatten')(self.last_output)
        self.x = tf.keras.layers.Dense(32, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.2)(self.x)
        self.x = tf.keras.layers.Dense(64, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.2)(self.x)
        self.x = tf.keras.layers.Dense(128, activation='relu')(self.x)
        self.x = tf.keras.layers.Dropout(0.5)(self.x)
        self.x = tf.keras.layers.Dense(2, activation='softmax')(self.x)

        self.model = tf.keras.models.Model(self.base_model.input, self.x)

        return self.model
    
    def compile(self, optimizer, loss, metrics):
        self.build_model().compile(optimizer=optimizer, loss=loss, metrics=metrics)