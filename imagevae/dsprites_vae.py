import torch
from torch import nn, distributions

from imagevae.mnist_vae import MnistVAE


class DspritesVAE(MnistVAE):
    def __init__(self):
        super(DspritesVAE, self).__init__()
        self.z_dim = 10             # z的维数
        self.inter_dim = 4
        self.enc_conv = nn.Sequential(      # 编码器卷积层
            nn.Conv2d(1, 32, 4, 2, 1),
            # (in_channels, out_channels, kernel_size, stride=2, padding=1)
            nn.ReLU(inplace=True),
            # inplace = False 时, 不会修改输入对象的值,而是返回一个新创建的对象,所以打印出对象存储地址不同,类似于C语言的值传递
            # inplace = True 时, 会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.enc_lin = nn.Sequential(      # 编码器全连接层
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.enc_mean = nn.Linear(256, self.z_dim)      # 编码器编码mean
        self.enc_log_std = nn.Linear(256, self.z_dim)      # 编码器编码log_std
        self.dec_lin = nn.Sequential(      # 解码器全连接层
            nn.Linear(self.z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )
        self.dec_conv = nn.Sequential(      # 解码器反卷积层
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # (in_channels, out_channels, kernel_size, stride=2, padding=1)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.xavier_initialization()        # 初始化

        self.update_filepath()      # 更新文件路径

    def __repr__(self):
        """
        String representation of class      类的字符串表示
        :return: string
        """
        return 'DspritesVAE' + self.trainer_config