import tensorflow as tf
from base.base_dataloader import BaseDataLoader
from tensorflow.keras.preprocessing import image_dataset_from_directory


class CornDataLoader(BaseDataLoader):
    def __init__(self, config, train_dir, val_dir, image_size, batch_size):
        super(CornDataLoader, self).__init__(config)
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir

    def get_trainloader(self):
        self.train_generator = image_dataset_from_directory(
            self.train_dir,
            label_mode='categorical',
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            validation_split=self.config.trainer.validation_split,
            seed=1337,
            subset='training',
            shuffle=True
        )
        return self.train_generator

    def get_validationloader(self):
        self.valid_generator = image_dataset_from_directory(
            self.val_dir,
            label_mode='categorical',
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            validation_split=self.config.trainer.validation_split,
            seed=1337,
            subset='validation',
            shuffle=True
        )
        return self.valid_generator
