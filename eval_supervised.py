from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

import torch
import torch.nn as nn

# from models import SupResNet, SSLResNet
from models1 import SupResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, labelstrain, args):
    # if args.clusters == 1:
    #     return get_scores_one_cluster(ftrain, ftest, food)
    return get_scores_multi_cluster(ftrain, ftest, food, labelstrain) 



def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def get_eval_results(ftrain, ftest, food, labelstrain, args):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, args)
    print("dtest size is:", dtest.shape)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr


def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument("--exp-name", type=str, default="temp_eval_ssd")
    # parser.add_argument(
    #     "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    # )
    parser.add_argument("--results-dir", type=str, default="./eval_results")

    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument(
        "--data-dir", type=str, default="./data/"
    )
    parser.add_argument(
        "--data-mode", type=str, choices=("org", "base", "ssl"), default="base"
    )
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=32)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    device = "cuda:0"

    assert args.ckpt, "Must provide a checkpint for evaluation"

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    results_file = os.path.join(args.results_dir, args.exp_name + "_supervised.txt")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(results_file, "a"))
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model

    model = SupResNet(arch=args.arch, num_classes=args.classes).eval()

    model.encoder = model.encoder.to(device)

    # # load checkpoint
    ckpt_dict = torch.load(args.ckpt, map_location="cpu")
    # if "model" in ckpt_dict.keys():
    #     ckpt_dict = ckpt_dict["model"]
    # if "net" in ckpt_dict.keys():
    #     ckpt_dict = ckpt_dict["net"]
    # model.load_state_dict(ckpt_dict)
    # print(ckpt_dict)
    ckpt_dict = ckpt_dict["net"]
    model.load_state_dict(ckpt_dict)

    # dataloaders
    train_loader, test_loader, norm_layer = data.__dict__[args.dataset](
        args.data_dir,
        args.batch_size,
        mode=args.data_mode,
        normalize=args.normalize,
        size=args.size,
    )

    features_train, labels_train = get_features(
        model.encoder, train_loader
    )  # using feature befor MLP-head

    np.save("./features/supervised_cifar10/features_train2.npy",features_train)
    np.save("./features/supervised_cifar10/labels_train2.npy",labels_train)
    print("features_train type is:", type(features_train))

    features_test, _ = get_features(model.encoder, test_loader)
    np.save("./features/supervised_cifar10/features_test2.npy",features_test)
    print("In-distribution features shape: ", features_train.shape, features_test.shape)

    # ds = ["cifar10", "cifar100", "svhn"]
    # ds = ["cifar10", "svhn"]
    ds = ["cifar10", "cifar100"]
    ds.remove(args.dataset)

    for d in ds:
        _, ood_loader, _ = data.__dict__[d](
            args.data_dir,
            args.batch_size,
            mode="base",
            normalize=args.normalize,
            norm_layer=norm_layer,
            size=args.size,
        )
        features_ood, _ = get_features(model.encoder, ood_loader)
        np.save("./features/supervised_cifar10/features_ood2.npy",features_ood)
        print("Out-of-distribution features shape: ", features_ood.shape)


        fpr95, auroc, aupr = get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            args,
        )

        logger.info(
            f"In-data = {args.dataset}, OOD = {d}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
        )


if __name__ == "__main__":
    main()
