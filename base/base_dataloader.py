class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_trainloader(self):
        raise NotImplementedError

    def get_validationloader(self):
        raise NotImplementedError
