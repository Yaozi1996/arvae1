import torch
from torch import nn, distributions

from utils.model import Model


class MnistVAE(Model):
    """
    Class defining a variational auto-encoder (VAE) for MNIST images
    类定义用于MNIST图像的变分自动编码器

    图片（batchsize，input_channels，高，宽）
    """
    def __init__(self):
        super(MnistVAE, self).__init__()
        self.input_size = 784
        self.z_dim = 16
        self.inter_dim = 19
        self.enc_conv = nn.Sequential(  #定义编码器卷积层
            nn.Conv2d(1, 64, 4, 1),
            # 输入厚度1，输出厚度64，卷积核大小4*4，步长1
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, 4, 1),
            # 输入厚度64，输出厚度64
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 8, 4, 1),
            # 输入厚度64，输出厚度8
            nn.SELU(),
            nn.Dropout(0.5),
        )
        """
        加Dropout可以解决过拟合：
        （1）减少神经元之间复杂的共适应关系：
        因此不能保证每2个隐含节点每次都同时出现
        这样权值的更新不再依赖于有固定关系隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况
        从这个角度看dropout就有点像L1，L2正则，减少权重使得网络对丢失特定神经元连接的鲁棒性提高
        （2）取平均的作用：
        dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同
        整个dropout过程就相当于对很多个不同的神经网络取平均
        而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。
        """
        self.enc_lin = nn.Sequential(  #定义编码器全连接层
            nn.Linear(2888, 256),
            nn.SELU()
        )
        self.enc_mean = nn.Linear(256, self.z_dim)  #定义编码中代表均值的z
        self.enc_log_std = nn.Linear(256, self.z_dim)  #定义编码中代表 log(标准差) 的z    （标准差的平方，是方差）

        self.dec_lin = nn.Sequential(  #定义解码器全连接层
            nn.Linear(self.z_dim, 256),
            nn.SELU(),
            nn.Linear(256, 2888),
            nn.SELU()
        )

        self.dec_conv = nn.Sequential(  #定义解码器卷积层
            nn.ConvTranspose2d(8, 64, 4, 1),    #反卷积
            # 输入厚度8，输出厚度64，卷积核大小4*4，步长1
            nn.SELU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 64, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 1, 4, 1),
        )
        self.xavier_initialization()    #定义初始化网络参数

        self.update_filepath()  # 定义更新文件路径

    def __repr__(self):
        """
        String representation of class  类的字符串表示
        :return: string
        """
        return 'MnistVAE' + self.trainer_config

    def encode(self, x):    # 定义编码器
        hidden = self.enc_conv(x)   # 卷积层处理
        hidden = hidden.view(x.size(0), -1)         # x：（BATCH_SIZE，out_channels，高，宽） --> hidden：（BATCH_SIZE，out_channels*高*宽）
        # 将卷积的输出拉伸为一行
        hidden = self.enc_lin(hidden)   # 全连接层处理

        z_mean = self.enc_mean(hidden)  # 编码z，代表均值
        z_log_std = self.enc_log_std(hidden)  # 编码z，代表log(标准差)
        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        # distributions.Normal(loc=0,scale=1) # 这个是标准正态分布

        return z_distribution

    def decode(self, z):
        hidden = self.dec_lin(z)
        hidden = hidden.view(z.size(0), -1, self.inter_dim, self.inter_dim)
        # input.size(0)表示第一维的大小，即batch_size，把每张图片的数据展开成一维
        # view(input.size(0), self.channel, self.height, self.width)
        hidden = self.dec_conv(hidden)
        return hidden

    def reparametrize(self, z_dist):
        """
        Implements the reparametrization trick for VAE （实现VAE的重新参数化技巧）
        """
        # sample from distribution 样本分布
        z_tilde = z_dist.rsample()
        """
        rsample():
        不是在定义的正太分布上采样，而是先对标准正太分布N(0,1)进行采样，然后输出：mean+std×采样值
        """

        # compute prior 计算之前
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            # 生成一个和z_dist.loc相同size的全零的张量
            scale=torch.ones_like(z_dist.scale)
            # 生成一个和z_dist.scale相同size的全一的张量
        )

        z_prior = prior_dist.sample()
        # 采样
        return z_tilde, z_prior, prior_dist

    def forward(self, x):
        """
        Implements the forward pass of the VAE （实现了VAE的向前传递）
        :param x: minist image input （参数x: minist图像输入）
            (batch_size, 28, 28)
        """
        # compute distribution using encoder 使用编码器计算分布
        z_dist = self.encode(x)

        # reparametrize 重新参数化
        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        # compute output of decoding layer 计算译码层的输出
        output = self.decode(z_tilde).view(x.size())

        return output, z_dist, prior_dist, z_tilde, z_prior

        """
        x = x.view(x.size(0), -1) 
        这句话一般出现在model类的forward函数中，具体位置一般都是在调用分类器之前。
        分类器是一个简单的nn.Linear()结构，输入输出都是维度为一的值
        x = x.view(x.size(0), -1)这句话的出现就是为了将前面多维度的tensor展平成一维。

        x.size(0)指batchsize的值
        x = x.view(x.size(0), -1)简化为x = x.view(batchsize, -1)。
        view()函数的功能根reshape类似，用来转换size大小
        x = x.view(batchsize, -1)中batchsize指转换后有几行
        而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        """