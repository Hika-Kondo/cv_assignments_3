import models
import solver
import dataloader
import preprocess

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import mlflow
import hydra
import omegaconf
import numpy as np

from logging import getLogger
from pathlib import Path
from collections import defaultdict
import random
import os


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)


def log_params(dic, name=None):
    for key, values in dic.items():
        if type(values) == omegaconf.dictconfig.DictConfig:
            if name is not None:
                key = name + "." + key
            log_params(values, key)
        else:
            if name is not None:
                key = name + "." + key
            mlflow.log_param(key, values)


@hydra.main("config.yaml")
def main(cfg):
    logger = getLogger(__name__)

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.ex_name)

    with mlflow.start_run() as run:
        log_params(cfg)

        generator = models.Generator()
        discriminator = models.Discriminator()
        logger.info(generator)
        logger.info(discriminator)

        train_transforms, val_transforms = preprocess.get_transform(**cfg.transforms.kwargs)
        train = dataloader.MyMNIST(root="../data", train=True,
                download=True, transform=train_transforms, limit_data=[0,cfg.num_data])


        train_dataloader = DataLoader(train, **cfg.dataloader.kwargs,)

        g_optim = getattr(torch.optim, cfg.optim.name)(generator.parameters(), **cfg.optim.kwargs)
        d_optim = getattr(torch.optim, cfg.optim.name)(discriminator.parameters(),
                **cfg.optim.kwargs)

        trainer= solver.Solver(generator, discriminator, train_dataloader,
                g_optim, d_optim, **cfg.solver.kwargs)

        res = trainer.train()

        for key, item in res.items():
            key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(key, item)

        trainer.save()

if __name__ == "__main__":
    main()
