
import os
import re
import numpy as np
import argparse
import logging
import time
import json
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-lightning"])

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

seed_everything(7)


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_array = self.images[idx].reshape(3, 32,32)
        image_array_new = np.transpose(image_array, (1, 2, 0))
        image = Image.fromarray(image_array_new).convert('RGB')

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        #print("image shape=", image.shape)
        return image, label	



class CIFAR100DataModule(pl.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """Data module class."""
    def __init__(self, **kwargs):
        """Initialization of inherited lightning data module."""
        super(CIFAR100DataModule, self).__init__()  # pylint: disable=super-with-arguments
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.label_train = None 
        self.label_valid = None
        self.label_test = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                              std=[0.267, 0.256, 0.276])
        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.args = kwargs

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def prepare_data(self):
        """Implementation of abstract class."""
        cifar10_data = torchvision.datasets.CIFAR100('.', download=True)
        metadata_path = './cifar-100-python/meta' 
        metadata = self.unpickle(metadata_path)
        superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))       
        data_pre_path = './cifar-100-python/' 
        # File paths
        data_train_path = data_pre_path + 'train'
        data_test_path = data_pre_path + 'test'
        # Read dictionary
        data_train_dict = self.unpickle(data_train_path)
        data_test_dict = self.unpickle(data_test_path)
        # Get data (change the coarse_labels if you want to use the 100 classes)
        d_train = data_train_dict[b'data']
        l_train = np.array(data_train_dict[b'coarse_labels'])
        self.data_train, self.data_valid, self.label_train, self.label_valid = train_test_split(d_train, l_train, test_size=0.2)

        self.data_test = data_test_dict[b'data']
        self.label_test = np.array(data_test_dict[b'coarse_labels'])


    def setup(self, stage=None):
        """Downloads the data, parse it and split the data into train, test,
        validation data.
        Args:
            stage: Stage - training or testing
        """

        self.train_dataset = CustomImageDataset(self.data_train, self.label_train, self.train_transform)
        self.valid_dataset = CustomImageDataset(self.data_valid, self.label_valid, self.valid_transform)
        self.test_dataset = CustomImageDataset(self.data_test, self.label_test, self.valid_transform)


    def create_data_loader(self, dataset, batch_size, num_workers):  # pylint: disable=no-self-use
        """Creates data loader."""
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers)

    def train_dataloader(self):
        """Train Data loader.
        Returns:
             output - Train data loader for the given input
        """
        if self.train_data_loader is None:
            self.train_data_loader = self.create_data_loader(
                self.train_dataset,
                self.args.get("batch_size", 64),
                self.args.get("num_workers", 4),
            )
        return self.train_data_loader

    def val_dataloader(self):
        """Validation Data Loader.
        Returns:
             output - Validation data loader for the given input
        """
        if self.val_data_loader is None:
            self.val_data_loader = self.create_data_loader(
                self.valid_dataset,
                self.args.get("batch_size", 64),
                self.args.get("num_workers", 4),
            )
        return self.val_data_loader

    def test_dataloader(self):
        """Test Data Loader.
        Returns:
             output - Test data loader for the given input
        """
        if self.test_data_loader is None:
            self.test_data_loader = self.create_data_loader(
                self.test_dataset,
                self.args.get("batch_size", 64),
                self.args.get("num_workers", 4),
            )
        return self.test_data_loader

def create_model():
    model_conv = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    num_classes = 100
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    return model_conv

class LitResnet(LightningModule):
    def __init__(self, batch_size, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        #self.model = create_model()
        self.model = create_model()

    def forward(self, x): 
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def train_cifar100(args):
    cifar100_dm = CIFAR100DataModule(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    #cifar100_dm.prepare_data()
    #cifar100_dm.setup()

    model = LitResnet(args.batch_size, lr=args.lr)
    model.datamodule = cifar100_dm

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        gpus=1,
        logger=TensorBoardLogger("lightning_logs/", name="resnet"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )


    trainer.fit(model, cifar100_dm)
    trainer.test(model, datamodule=cifar100_dm)

    trainer.save_checkpoint(args.checkpoint_path + '/checkpoint.pth')
    print(f'Model saved to {args.checkpoint_path}')

    return _save_model(model, args.model_dir)   

def _save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html    
    torch.save(model.cpu().state_dict(), path)
    print(f'Model saved to :{path}')

def _save_checkpoint(model, optimizer, epoch, loss, args):
    print(f'epoch : {epoch+1}, loss : {loss}')
    checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
    print(f'saving the checkpoint : {checkpointing_path}')
    torch.save({
        'epoch':epoch+1,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
    }, checkpointing_path)
    
# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def _load_checkpoint(model, optimizer, args):
    print('-------')
    print('checkpoint file found !')
    print(f"loading checkpoint from {args.checkpoint_path}'/checkpoint.pth'")
    checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_number = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'checkpoint file loaded, epoch_number : {epoch_number} & loss : {loss}')
    print(f'Resuming training from epoch : {epoch_number + 1}')
    print('-------')
    return model, optimizer, epoch_number

def model_fn(model_dir):
    print("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    model = LitResnet(batch_size)
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        print(f'Model Loaded successfully from {model_dir}')
    return model.to(device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="BS", help="batch size (default: 4)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument("--current-host", type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--num-gpus", type=int, default=1)
    
    args, _ = parser.parse_known_args()
    print('Printing Args :', args)

    train_cifar100(parser.parse_args())      
