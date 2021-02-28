import os
import time
import datetime
from tqdm import tqdm

from abc import ABC, abstractmethod
import torch
from torch import nn

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.helpers import to_numpy


class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    将作为 NN trainer 的抽象基类
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4):
        """
        Initializes the trainer class
        初始化训练的类
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.global_iter = 0
        self.trainer_config = ''
        self.writer = None

    def train_model(self, batch_size, num_epochs, log=False):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None

        训练模型
        :参数 batch_size:   int型，
        :参数 num_epochs:   int型，
        :参数 log:          bool型，如果为真，记录轮次统计数据，以便在tensorboard中查看，
        返回: 无
        """
        # set-up log parameters
        # 设置日志参数
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboardX summary writer
            # 配置tensorboardX摘要编写器
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + self.model.__repr__() + st)
            )

        # get dataloaders
        # 获取数据加载器
        (generator_train,
         generator_val,
         _) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        # 训练轮
        for epoch_index in range(num_epochs):
            # update training scheduler
            # 更新训练计划
            self.update_scheduler(epoch_index)

            # run training loop on training data
            # 对训练数据进行训练循环
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True
            )

            # run evaluation loop on validation data
            # 对验证数据运行评估循环
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False
            )

            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index,
            )

            # log parameters
            # 日志参数
            if log:
                # log value in tensorboardX for visualization
                # tensorboardX中的日志值以进行可视化
                self.writer.add_scalar('loss/train', mean_loss_train, epoch_index)
                self.writer.add_scalar('loss/valid', mean_loss_val, epoch_index)
                self.writer.add_scalar('acc/train', mean_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/valid', mean_accuracy_val, epoch_index)

            # print epoch stats
            # 打印轮数统计
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            self.print_epoch_stats(**data_element)

            # save model
            # 保存模型
            self.model.save()

    # 计算每轮的损失和准确性
    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        计算一个轮的损失和准确性
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            # process batch data
            # 处理批次数据
            batch_data = self.process_batch_data(batch)

            # zero the gradients
            # 梯度清零
            self.zero_grad()

            # compute loss for batch
            # 计算批次损失
            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, batch_num, train=train
            )

            # compute backward and step if train
            # 若训练 则计算反向传播和步调
            if train:
                loss.backward()
                # self.plot_grad_flow()
                self.step()

            # compute mean loss and accuracy
            # 计算平均损失和准确性
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    # 将模型转换为cuda
    def cuda(self):
        """
        Convert the model to cuda
        """
        self.model.cuda()

    # 将相关优化器的梯度归零
    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.optimizer.zero_grad()

    # 对所有优化器执行向后传递和逐步更新
    def step(self):
        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
        self.optimizer.step()

    # 评估模型
    def eval_model(self, data_loader, epoch_num):
        """
        This can contain any method to evaluate the performance of the mode
        Possibly add more things to the summary writer
        它可以包含任何评估模式性能的方法
        可能向摘要编写器添加更多内容
        """
        pass

    # 下载模型
    def load_model(self):
        is_cpu = False if torch.cuda.is_available() else True
        self.model.load(cpu=is_cpu)
        if not is_cpu:
            self.model.cuda()

    @abstractmethod
    # 计算每个批次的损失和精度
    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        计算批次的损失和准确性
        必须以元组形式返回（（损失，准确性），准确性可以为None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int,
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        pass

    @abstractmethod
    # 处理批次数据
    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        处理由数据加载器迭代器返回的批处理
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        pass

    # 更新调度器
    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        pass

    @staticmethod
    # 打印轮数统计
    def print_epoch_stats(
            epoch_index,
            num_epochs,
            mean_loss_train,
            mean_accuracy_train,
            mean_loss_val,
            mean_accuracy_val
    ):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )

    @staticmethod
    # 平均交叉熵损失
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        评估交叉熵损失
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    # 平均准确度
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        评估预测的平均准确性
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)

        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    # 均值L1损失RNN
    def mean_l1_loss_rnn(weights, targets):
        """
        Evaluates the mean l1 loss
        评估均值L1损失
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.L1Loss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    # 均值MSE损失RNN
    def mean_mse_loss_rnn(weights, targets):
        """
        Evaluates the mean mse loss
        评估均值MSE损失
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.MSELoss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    # 均值交叉熵损失
    def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        评估交叉熵损失
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    # 均值精确度
    def mean_accuracy_alt(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        评估预测的平均准确性
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return float, accuracy
        """
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    # 计算KLD损失
    def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
        """
        :param z_dist: torch.distributions object
        :param prior_dist: torch.distributions
        :param beta: weight for kld loss
        :param c: capacity of bottleneck channel    瓶颈通道容量
        :return: kl divergence loss
        """
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = kld.sum(1).mean()
        kld = beta * (kld - c).abs()
        return kld

    @staticmethod
    # 计算正则化损失
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        """
        Computes the regularization loss
        计算正则化损失
        """
        x = z[:, reg_dim]
        # x取z的某一个正则化维度
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss
        # 最终的正则化损失部分为：gamma * reg_loss

    @staticmethod
    # 正则化损失
    def reg_loss_sign(latent_code, attribute, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        给定潜在代码和属性，计算正则化损失
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss  正则化维数距离矩阵的系数参数
        Returns
            scalar, loss 标量，损失
        """

        # 计算潜在距离矩阵
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)   # 得到正则化维数距离矩阵

        # repeat()： 在 latent_code.shape[0] 维度重复 1 次

        # 计算属性距离矩阵
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)   # 得到属性距离矩阵

        # 计算正则化损失
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss
        # Lr,a = MAE(tanh(δDr) - sgn(Da))

    @staticmethod
    # 获取保存路径
    def get_save_dir(model, sub_dir_name='results'):
        path = os.path.join(
            os.path.dirname(model.filepath),
            sub_dir_name
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path