import torch

class TrainGlobalConfig:
    num_workers = 0
    batch_size = 1
    n_epochs = 3
    lr = 0.0001

    folder = '../Model_Save(Axial)_D7'
    verbose = True
    verbose_step = 1
    step_scheduler = False  
    epoch_scheduler = False
    validation_scheduler = True

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.1,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0,
        eps=1e-08
    )
    
