import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from data.dataloaders.morphomnist import io, morpho


class MnistDataset:
    def __init__(self):
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.root_dir = os.path.join(               # root_dir表示获取根目录，获取项目名称
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'mnist_data'
        )
        self.train_dataset = datasets.MNIST(
            self.root_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.val_dataset = datasets.MNIST(
            self.root_dir, train=False, download=True, transform=transforms.ToTensor()
        )

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        train_dl = DataLoader(  # 训练集
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )
        val_dl = DataLoader(    # 验证集
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        eval_dl = DataLoader(   # 测试集
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_dl, val_dl, eval_dl


"""
python中有些常见方法参数是：
*self._args, **kwargs，如：self._target(*self._args, **self._kwargs)。
经过查找一些资料，可以归纳为以下两种类型：
1.*self._args  表示接受元组类参数；
2.**kwargs  表示接受字典类参数
"""

class MorphoMnistDataset(MnistDataset):
    def __init__(self):
        super(MorphoMnistDataset, self).__init__()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        # num_worker是工作进程数
        # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，
        # 这样将内存的Tensor转到GPU的显存就会更快一些
        # 计算机的内存充足的时候，可以设置pin_memory=True。
        # 当系统卡住，或者交换内存使用过多的时候，设pin_memory=False。
        self.root_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'mnist_data',
            'plain'
        )

        self.data_path_str = "-images-idx3-ubyte.gz"
        self.label_path_str = "-labels-idx1-ubyte.gz"
        self.morpho_path_str = "-morpho.csv"

        self.train_dataset = self._create_dataset(dataset_type="train")
        self.val_dataset = self._create_dataset(dataset_type="t10k")

    def _create_dataset(self, dataset_type="train"):
        data_path = os.path.join(
            self.root_dir,
            dataset_type + self.data_path_str
        )
        label_path = os.path.join(
            self.root_dir,
            dataset_type + self.label_path_str
        )
        morpho_path = os.path.join(
            self.root_dir,
            dataset_type + self.morpho_path_str
        )

        images = io.load_idx(data_path)
        images = np.expand_dims(images, axis=1).astype('float32') / 255.0
        labels = io.load_idx(label_path)
        morpho_labels = pd.read_csv(morpho_path).values.astype('float32')

        dataset = TensorDataset(
            torch.from_numpy(images),
            torch.from_numpy(labels),
            torch.from_numpy(morpho_labels)
        )
        return dataset
