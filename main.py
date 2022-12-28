from dataloader.corn_dataloader import CornDataLoader
from models.corn_classifier import CornDiseaseClassifier
from trainers.corn_recog_trainer import CornDiseaseRecognitionTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = CornDataLoader(config, 'data/', 'data/', (224, 224), config.trainer.batch_size)

    print('Create the model.')
    model = CornDiseaseClassifier(config)

    print('Create the trainer')
    trainer = CornDiseaseRecognitionTrainer(model.model, data_loader.get_trainloader(), data_loader.get_validationloader(), config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
