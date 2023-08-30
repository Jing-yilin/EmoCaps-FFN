"""use GAN-FFN to train iemocap_data to the feature fusion data
"""

import os
import sys

import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from models import (
    MaskedNLLLoss,
    FocalLoss,
    LSTMModel2,
    AcousticGenerator,
    AcousticDiscriminator,
    TextGenerator,
    TextDiscriminator,
    VisualGenerator,
    VisualDiscriminator,
    GAN_FFN,
)
from dataload import IEMOCAPDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import random

import warnings

warnings.filterwarnings("ignore")

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seed(seed: int = 3407) -> None:
    """seed = 3407
    https://arxiv.org/pdf/2109.08203.pdf
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(
    path, batch_size=32, valid=0.2, num_workers=0, pin_memory=False
):
    trainset = IEMOCAPDataset(path=path, train=True)
    testset = IEMOCAPDataset(path=path, train=False)

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def train_or_eval_model(
    model, loss_function, dataloader, epoch, optimizer=None, train=False, cuda=True
):
    """
    Utility function to train model for one epoch of train data
    or evaluate model on val/test data.
    :param model: torch NN model
    :param loss_function: loss function to optimize
    :param dataloader: torch Dataloader for train/val/test data
    :param epoch: number of epoch
    :param optimizer: optimizer to use to train model
    :param train: boolean value if train or val/test
    """
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            lambda1 = lambda epoch: 0.98**epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda1
            )  # 打印学习率，防止过拟合
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = (
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        )
        seq_lengths = [
            (umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))
        ]

        # print("textf.shape = ", textf.shape) # (seq_len, 32, 100)
        # print("visuf.shape = ", visuf.shape) # (seq_len, 32, 512)
        # print("acouf.shape = ", acouf.shape) # (seq_len, 32, 100)
        # print("qmask.shape = ", qmask.shape) # (seq_len, 32, 2) 可以看出是哪一个人说的
        # print("umask.shape = ", umask.shape) # (32, seq_len) 可以看出每个句子是否有效 还是为0填充
        # print("label.shape = ", label.shape) # (32, seq_len)
        # print(label)
        # sys.exit()

        log_prob, alpha, alpha_f, alpha_b = model(acouf, visuf, textf)
        # log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf), dim=-1), qmask, umask)
        # log_prob, alpha, alpha_f, alpha_b, hidden = model(textf, acouf, visuf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            scheduler.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float("nan"), float("nan"), [], [], [], float("nan"), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(
        f1_score(labels, preds, sample_weight=masks, average="weighted") * 100, 2
    )
    return (
        avg_loss,
        avg_accuracy,
        labels,
        preds,
        masks,
        avg_fscore,
        [alphas, alphas_f, alphas_b, vids],
    )


def train_disc(
    disc, real_dics, gen, real_gen, opt, adversarial_loss, valid, fake
) -> float:
    """
    Train discriminator
    Args:
        disc: discriminator
        real_dics: real data input to discriminator
        gen: generator
        real_gen: real data input to generator
        opt: optimizer
        adversarial_loss: adversarial loss
        valid: valid label
        fake: fake label
    """
    disc.train()
    gen.eval()

    opt.zero_grad()
    real_prob = disc(real_dics)
    fusion = gen(real_gen)
    fake_prob = disc(fusion.detach())
    d_loss = (
        adversarial_loss(real_prob, valid) + adversarial_loss(fake_prob, fake)
    ) / 2.0
    res = d_loss.cpu().detach().numpy()
    d_loss.backward()
    opt.step()
    return res


def train_gen(gen, real_gen, disc, opt, adversarial_loss, valid, fake) -> float:
    """
    Train generator
    Args:
        gen: generator
        real_gen: real data input to generator
        disc: discriminator
        opt: optimizer
        adversarial_loss: adversarial loss
        valid: valid label
        fake: fake label
    """
    gen.train()
    disc.eval()

    opt.zero_grad()
    fusion = gen(real_gen)
    prob = disc(fusion)
    g_loss = adversarial_loss(prob, valid)
    res = g_loss.cpu().detach().numpy()
    g_loss.backward()
    opt.step()
    return res


def train_GAN(
    acoustic_gen: AcousticGenerator,
    visual_gen: VisualGenerator,
    text_gen: TextGenerator,
    acoustic_disc: AcousticDiscriminator,
    visual_disc: VisualDiscriminator,
    text_disc: TextDiscriminator,
    epochs=50,
    batch_size=32,
    lr=0.002,
    b1=0.6,
    b2=0.996,
    dataset_path="./data/iemocap/IEMOCAP_features.pkl",
) -> pd.DataFrame:
    """
    Train the GAN model
    :param acoustic_gen: acoustic generator model
    :param visual_gen: visual generator model
    :param text_gen: text generator model
    :param acoustic_disc: acoustic discriminator model
    :param visual_disc: visual discriminator model
    :param text_disc: text discriminator model
    :param epochs: number of epochs to train
    :param batch_size: batch size
    :param lr: learning rate
    :param b1: Adam optimizer beta1
    :param b2: Adam optimizer beta2
    :param dataset_path: path to dataset

    :return: pandas dataframe with training information
    """
    # ----------
    #  Training
    # ----------
    print("=" * 15, "start training GAN", "=" * 15)

    # Optimizers
    opt_acoustic_G = torch.optim.Adam(acoustic_gen.parameters(), lr=lr, betas=(b1, b2))
    opt_acoustic_D = torch.optim.Adam(
        acoustic_disc.parameters(), lr=lr / 2, betas=(b1, b2)
    )
    opt_visual_G = torch.optim.Adam(visual_gen.parameters(), lr=lr, betas=(b1, b2))
    opt_visual_D = torch.optim.Adam(visual_disc.parameters(), lr=lr / 2, betas=(b1, b2))
    opt_text_G = torch.optim.Adam(text_gen.parameters(), lr=lr * 1.1, betas=(b1, b2))
    opt_text_D = torch.optim.Adam(text_disc.parameters(), lr=lr / 2, betas=(b1, b2))

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()  # 二元交叉熵

    # Dataloaders
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
        dataset_path, batch_size=batch_size, valid=0.1
    )
    # 新建一个df用于存储训练过程中的loss
    # Acoustic G D D, Visual G D D, Text G D D
    columns = [
        "epoch",
        "acoustic_G_loss",
        "visual_G_loss",
        "text_G_loss",
        "visual_D_loss",
        "text_D_loss",
        "acoustic_D_loss",
    ]
    loss_df = pd.DataFrame(columns=columns)

    # Start epochs
    for epoch in range(epochs):
        print("=" * 15, "start Epoch : ", epoch, "=" * 15)
        for i, data in enumerate(train_loader):
            loss = {
                "epoch": epoch,
                "acoustic_G_loss": 0,
                "visual_G_loss": 0,
                "text_G_loss": 0,
                "visual_D_loss": 0,
                "text_D_loss": 0,
                "acoustic_D_loss": 0,
            }

            textf, visuf, acouf, qmask, umask, label = (
                [d.to(device) for d in data[:-1]] if device else data[:-1]
            )

            batch_size = textf.size(1)
            seq_len = textf.size(0)

            # Adversarial ground truths
            valid = Variable(
                FloatTensor(seq_len, batch_size, 1).fill_(1.0), requires_grad=False
            ).to(device)
            fake = Variable(
                FloatTensor(seq_len, batch_size, 1).fill_(0.0), requires_grad=False
            ).to(device)

            # Configure input
            real_text = Variable(textf.type(FloatTensor))
            real_visual = Variable(visuf.type(FloatTensor))
            real_acoustic = Variable(acouf.type(FloatTensor))
            label = Variable(label.type(LongTensor))

            #  VisualDiscriminator vs AcousticGenerator
            loss["visual_D_loss"] = train_disc(
                visual_disc,
                real_visual,
                acoustic_gen,
                real_acoustic,
                opt_visual_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  AcousticGenerator vs VisualDiscriminator
            loss["acoustic_G_loss"] = train_gen(
                acoustic_gen,
                real_acoustic,
                visual_disc,
                opt_acoustic_G,
                adversarial_loss,
                valid,
                fake,
            )

            #  VisualDiscriminator vs TextGenerator
            loss["visual_D_loss"] = train_disc(
                visual_disc,
                real_visual,
                text_gen,
                real_text,
                opt_visual_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  TextGenerator vs VisualDiscriminator
            loss["text_G_loss"] = train_gen(
                text_gen,
                real_text,
                visual_disc,
                opt_text_G,
                adversarial_loss,
                valid,
                fake,
            )

            #  TextDiscriminator vs AcousticGenerator
            loss["text_D_loss"] = train_disc(
                text_disc,
                real_text,
                acoustic_gen,
                real_acoustic,
                opt_text_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  AcousticGenerator vs TextDiscriminator
            loss["acoustic_G_loss"] = train_gen(
                acoustic_gen,
                real_acoustic,
                text_disc,
                opt_acoustic_G,
                adversarial_loss,
                valid,
                fake,
            )

            #  AcousticDiscriminator vs TextGenerator
            loss["acoustic_D_loss"] = train_disc(
                acoustic_disc,
                real_acoustic,
                text_gen,
                real_text,
                opt_acoustic_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  TextGenerator vs AcousticDiscriminator
            loss["text_G_loss"] = train_gen(
                text_gen,
                real_text,
                acoustic_disc,
                opt_text_G,
                adversarial_loss,
                valid,
                fake,
            )

            #  TextDiscriminator vs VisualGenerator
            loss["text_D_loss"] = train_disc(
                text_disc,
                real_text,
                visual_gen,
                real_visual,
                opt_text_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  VisualGenerator vs TextDiscriminator
            loss["visual_G_loss"] = train_gen(
                visual_gen,
                real_visual,
                text_disc,
                opt_visual_G,
                adversarial_loss,
                valid,
                fake,
            )

            #  AcousticDiscriminator vs VisualGenerator
            loss["acoustic_D_loss"] = train_disc(
                acoustic_disc,
                real_acoustic,
                visual_gen,
                real_visual,
                opt_acoustic_D,
                adversarial_loss,
                valid,
                fake,
            )
            #  VisualGenerator vs AcousticDiscriminator
            loss["visual_G_loss"] = train_gen(
                visual_gen,
                real_visual,
                acoustic_disc,
                opt_visual_G,
                adversarial_loss,
                valid,
                fake,
            )

            # 以表格的形式打印loss这个字典
            loss = pd.DataFrame(loss, index=[0])
            print(loss)

            # 在每个epoch的最后一次batch中，将loss添加到loss_df中
            # 把loss添加到loss_df中
            if i == len(train_loader) - 1:
                # loss_df添加上loss
                loss_df = pd.concat([loss_df, loss], axis=0, ignore_index=True)
    return loss_df


def create_path(path: str) -> None:
    """创建路径"""
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)


def draw_GAN_loss(df: pd.DataFrame, path="./output/GAN_loss.png") -> None:
    """画出GAN的loss曲线"""
    plt.figure(figsize=(10, 8), dpi=300)
    # 画出loss曲线
    plt.plot(df["epoch"], df["acoustic_G_loss"], label="acoustic_G_loss")
    plt.plot(df["epoch"], df["visual_G_loss"], label="visual_G_loss")
    plt.plot(df["epoch"], df["text_G_loss"], label="text_G_loss")
    plt.plot(df["epoch"], df["visual_D_loss"], label="visual_D_loss")
    plt.plot(df["epoch"], df["text_D_loss"], label="text_D_loss")
    plt.plot(df["epoch"], df["acoustic_D_loss"], label="acoustic_D_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("GAN loss")
    # 保存
    create_path(path)
    plt.savefig(path)


def save_GAN_loss(df: pd.DataFrame, path="./output/GAN_loss.csv") -> None:
    create_path(path)
    df.to_csv(path, index=False)


def save_GAN_models(models: list, save_path: str) -> None:
    models_name = [
        "acoustic_gen",
        "acoustic_disc",
        "visual_gen",
        "visual_disc",
        "text_gen",
        "text_disc",
    ]
    for id, model in enumerate(models):
        path = save_path + models_name[id] + ".pth"
        torch.save(model, path)


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4027, help="random seed")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="does not use GPU"
    )
    # b1
    parser.add_argument(
        "--b1", type=float, default=0.6, metavar="B1", help="adam: decay of first order momentum of gradient"
    )
    # b2
    parser.add_argument(
        "--b2", type=float, default=0.996, metavar="B2", help="adam: decay of first order momentum of gradient"
    )    
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="LR", help="learning rate"
    )
    parser.add_argument(
        "--l2", type=float, default=0.008, metavar="L2", help="L2 regularization weight"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.6, metavar="dropout", help="dropout rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="BS", help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=160, metavar="E", help="number of epochs"
    )
    parser.add_argument(
        "--GAN-epochs", type=int, default=50, metavar="E", help="number of GAN epochs"
    )
    parser.add_argument(
        "--class-weight", action="store_true", default=True, help="use class weight"
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        default=False,
        help="use attention on top of lstm",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enables tensorboard log",
    )

    return parser.parse_args()


if __name__ == "__main__":
    dataset_path = "./iemocap_data.pkl"
    model_save_path = "./GAN_save/"
    # 创建文件
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    args = get_parse_args()
    print(args)

    set_all_seed(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print("Running on GPU")
    else:
        print("Running on CPU")

    path = "./tensorboard"

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(path)

    batch_size = args.batch_size
    cuda = args.cuda
    n_epochs = args.epochs
    dropout = args.dropout
    b1 = args.b1
    b2 = args.b2
    lr = args.lr
    l2 = args.l2
    g_epochs = args.GAN_epochs

    n_classes = 6
    D_m = 200
    D_e = 30
    D_h = 100
    text_features = 100
    visual_features = 256
    acoustic_features = 100

    # create GAN components
    acoustic_gen = AcousticGenerator(D_h, features=acoustic_features, dropout=dropout)
    acoustic_disc = AcousticDiscriminator(D_h, features=acoustic_features, dropout=dropout)
    visual_gen = VisualGenerator(D_h, features=visual_features, dropout=dropout)
    visual_disc = VisualDiscriminator(D_h, features=visual_features, dropout=dropout)
    text_gen = TextGenerator(D_h, features=text_features, dropout=dropout)
    text_disc = TextDiscriminator(D_h, features=text_features, dropout=dropout)

    if cuda:
        acoustic_gen = nn.DataParallel(acoustic_gen).cuda()
        acoustic_disc = nn.DataParallel(acoustic_disc).cuda()
        visual_gen = nn.DataParallel(visual_gen).cuda()
        visual_disc = nn.DataParallel(visual_disc).cuda()
        text_gen = nn.DataParallel(text_gen).cuda()
        text_disc = nn.DataParallel(text_disc).cuda()

    loss_df = train_GAN(
        acoustic_gen,
        visual_gen,
        text_gen,
        acoustic_disc,
        visual_disc,
        text_disc,
        epochs=g_epochs,
        batch_size=batch_size,
        lr=lr,
        b1=b1,
        b2=b2,
        dataset_path=dataset_path,
    )

    save_GAN_loss(loss_df, f"./output/GAN_loss.csv")
    draw_GAN_loss(loss_df, f"./output/GAN_loss.png")
    # save model parameters
    models = [
        acoustic_gen,
        acoustic_disc,
        visual_gen,
        visual_disc,
        text_gen,
        text_disc,
    ]
    save_GAN_models(models, model_save_path)
