import os
import click
import torch
import json
from data.dataloaders.mnist_dataset import MorphoMnistDataset
from data.dataloaders.dsprites_dataset import DspritesDataset
from imagevae.mnist_vae import MnistVAE
from imagevae.dsprites_vae import DspritesVAE
from imagevae.image_vae_trainer import ImageVAETrainer, MNIST_REG_TYPES, DSPRITES_REG_TYPE

"""
Click 是用 Python 写的一个第三方模块，用于快速创建命令行。
我们知道，Python 内置了一个 argparse 的标准库用于创建命令行，但使用起来有些繁琐，Click 相比于 argparse.
Click 的使用大致有两个步骤：
1.使用 @click.command() 装饰一个函数，使之成为命令行接口；
2.使用 @click.option() 等装饰函数，为其添加命令行选项等。
"""


@click.command()
@click.option('--dataset_type', '-d', default='mnist',
              help='dataset to be used, `mnist` or `dsprites`')  # 数据集被使用，' mnist '或' dsprites
@click.option('--batch_size', default=128,
              help='training batch size')   # 训练批大小
@click.option('--num_epochs', default=100,
              help='number of training epochs') # 训练轮数大小
@click.option('--lr', default=1e-4,
              help='learning rate') # 学习率
@click.option('--beta', default=4.0,
              help='parameter for weighting KLD loss')  # KLD loss的加权参数
@click.option('--capacity', default=0.0,
              help='parameter for beta-VAE capacity')   # β-VAE的容量参数
@click.option('--gamma', default=10.0,
              help='parameter for weighting regularization loss')   # 正则化损失的加权参数
@click.option('--delta', default=1.0,
              help='parameter for controlling the spread')  # 参数，用于控制扩展
@click.option('--dec_dist', default='bernoulli',
              help='distribution of the decoder')   # 解码器的分布
@click.option('--train/--test', default=True,
              help='train or test the specified model') # 训练或测试指定的模型
@click.option('--log/--no_log', default=False,
              help='log the results for tensorboard')   # 记录张量板的结果
@click.option(
    '--rand',
    default=None,
    help='random seed for the random number generator'  # 用于随机数生成器的随机种子
)
@click.option(
    '--reg_type',
    '-r',
    default=None,
    multiple=True,
    help='attribute name string to be used for regularization'  # 用于正则化的属性名称字符串
)
def main(
        dataset_type,       # 数据集被使用，' mnist '或' dsprites '
        batch_size,         # 训练批大小
        num_epochs,         # 训练轮数大小
        lr,                 # 学习率
        beta,               # KLD loss的加权参数
        capacity,           # β-VAE的容量参数
        gamma,              # 正则化损失的加权参数
        delta,              # 参数，用于控制扩展
        dec_dist,           # 解码器的分布
        train,              # 训练或测试指定的模型
        log,                # 记录张量板的结果
        rand,               # 用于随机数生成器的随机种子
        reg_type,           # 用于正则化的属性名称字符串
):
    if dataset_type == 'mnist':
        dataset = MorphoMnistDataset()
        model = MnistVAE()
        attr_dict = MNIST_REG_TYPES
    elif dataset_type == 'dsprites':
        dataset = DspritesDataset()
        model = DspritesVAE()
        attr_dict = DSPRITES_REG_TYPE
    else:
        raise ValueError("Invalid dataset_type. Choose between mnist and dsprites")
    
    # 把属性传进来
    if len(reg_type) != 0:          # 用于正则化的属性名称长度是否为零
        if len(reg_type) == 1:
            if reg_type[0] == 'all':
                reg_dim = []
                for r in attr_dict.keys():
                    if r == 'digit_identity' or r == 'color':
                        continue
                    reg_dim.append(attr_dict[r])
            else:
                reg_dim = [attr_dict[reg_type]]
        else:
            reg_dim = []
            for r in reg_type:
                reg_dim.append(attr_dict[r])
    else:
        reg_dim = [0]
    reg_dim = tuple(reg_dim)

    if rand is None:
        rand = range(0, 10)
    else:
        rand = [int(rand)]
    for r in rand:
        # instantiate trainer   实例化训练
        trainer = ImageVAETrainer(
            dataset=dataset,
            model=model,
            lr=lr,
            reg_type=reg_type,
            reg_dim=reg_dim,
            beta=beta,
            capacity=capacity,
            gamma=gamma,
            delta=delta,
            dec_dist=dec_dist,
            rand=r
        )

        # train if needed   如果需要训练的话
        if train:
            if torch.cuda.is_available():
                trainer.cuda()
    
            trainer.train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                log=log
            )

        # compute and print evaluation metrics  计算和打印评估指标
        trainer.load_model()
        trainer.writer = None
        metrics = trainer.compute_eval_metrics()    # 计算评价指标
        print(json.dumps(metrics, indent=2))        # 缩进为2

        for sample_id in [0, 1, 4]:
            trainer.create_latent_gifs(sample_id=sample_id)

        # interp_dict = metrics['interpretability']
        # if dataset_type == 'mnist':
        #     attr_dims = [interp_dict[attr][0] for attr in trainer.attr_dict.keys() if attr != 'digit_identity']
        #     non_attr_dims = [a for a in range(trainer.model.z_dim) if a not in attr_dims]
        #     for attr in interp_dict.keys():
        #         dim1 = interp_dict[attr][0]
        #         trainer.plot_latent_surface(
        #             attr,
        #             dim1=dim1,
        #             dim2=non_attr_dims[-1],
        #             grid_res=0.05,
        #         )

        # # plot interpolations
        # trainer.plot_latent_reconstructions()
        # for attr_str in trainer.attr_dict.keys():
        #     if attr_str == 'digit_identity' or attr_str == 'color':
        #         continue
        #     trainer.plot_latent_interpolations(attr_str)

        # if dataset_type == 'mnist':
        #     trainer.plot_latent_interpolations2d('slant', 'thickness')
        # else:
        #     trainer.plot_latent_interpolations2d('posx', 'posy')


if __name__ == '__main__':
    main()
