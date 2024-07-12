import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from .train_global_config import TrainGlobalConfig
from project.utils.collate import collate_fn
from project.model.fitter import Fitter

class Model(): #proxy
    pass


def run_training(train_dataset, val_dataset):
    # need to set a class/function for this Model
    net = Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=False,  # drop last one for having same batch size
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history = fitter.fit(
        train_loader, val_loader)

    return best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history