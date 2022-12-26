from base.base_trainer import BaseTrain

class CornDiseaseRecognitionTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)