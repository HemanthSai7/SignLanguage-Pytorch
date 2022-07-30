import torch

#Configuration
class Config:
    output=9
    num_workers=4
    batch_size=32

    img_size=224
    n_epochs=200
    lr=0.0003
    patience=5

    SchedulerClass=torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params=dict(
        mode='min',
        factor=0.8,
        patience=1,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )