import tensorflow as tf
from base.base_dataloader import BaseDataLoader


class CornDataLoader(BaseDataLoader):
    def __init__(self, config, train_dir, val_dir, image_size, batch_size):
        super(CornDataLoader, self).__init__(config)
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255,
            validation_split=self.config.trainer.validation_split,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255,
            validation_split=self.config.trainer.validation_split    
        )

    def get_trainloader(self):
        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            subset='training',
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        return self.train_generator

    def get_validationloader(self):
        self.valid_generator = self.valid_datagen.flow_from_directory(
            self.val_dir,
            subset='validation',
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        return self.valid_generator
