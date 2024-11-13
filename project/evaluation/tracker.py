import pandas as pd

class Tracker():

    __slots__ = ['date', 'model', 'loaded_weights', 'uses_resnet', 'only_cmb_slices', 'cohort1', 'cohort2', 'cohort3', 'optimizer', 'epochs', 'loss', 'lr', 'saved_weights', 'saved_thist', 'saved_vhist', 'model_hyperparams', 'logfile', 'device', 'iou', 'dice', 'precision', 'recall', 'f1', 'fpr', 'batch_size', 'test_size', 'target_shape', 'stage1_weights']

    def __init__(self):
        self.date=None
        self.model=None 
        self.loaded_weights=None 
        self.uses_resnet=None 
        self.only_cmb_slices=None 
        self.cohort1=False
        self.cohort2=False
        self.cohort3=False
        self.optimizer=None 
        self.epochs=None 
        self.loss=None 
        self.lr=None 
        self.saved_weights=None 
        self.saved_thist=None 
        self.saved_vhist=None 
        self.model_hyperparams=None 
        self.logfile=None 
        self.device=None
        self.iou=None
        self.dice=None
        self.precision=None
        self.recall=None
        self.f1=None
        self.fpr=None
        self.batch_size=None
        self.test_size=None
        self.target_shape=None
        self.stage1_weights=None

    def __call__(self):
        # This method to ensure this order is showed in the df
        df = pd.DataFrame(
            dict(
                date = [self.date],
                model = [self.model],
                loaded_weights = [self.loaded_weights],
                stage1_weights = [self.stage1_weights],
                uses_resnet = [self.uses_resnet],
                only_cmb_slices = [self.only_cmb_slices],
                cohort1 = [self.cohort1],
                cohort2 = [self.cohort2],
                cohort3 = [self.cohort3],
                optimizer = [self.optimizer],
                epochs = [self.epochs],
                batch_size = [self.batch_size],
                test_size = [self.test_size],
                loss = [self.loss],
                lr = [self.lr],
                target_shape = [self.target_shape],
                iou = [self.iou],
                dice = [self.dice],
                precision = [self.precision],
                recall = [self.recall],
                f1 = [self.f1],
                fpr = [self.fpr],
                saved_weights = [self.saved_weights],
                saved_thist = [self.saved_thist],
                saved_vhist = [self.saved_vhist],
                model_hyperparams = [self.model_hyperparams],
                logfile = [self.logfile],
                device = [self.device],
            )
        )

        df = df.set_index('date')

        return df

    def nca(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith('__')]

    def missing(self):
        atts = self.nca()
        print('Missing information:')
        for i in atts:
            if getattr(self, i) == None:
                print(i)
