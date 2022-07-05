from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

from functools import partial

import torch
import torch.nn as nn

from models import SupResNet, SSLResNet
from utils import (
    # get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data


def get_features_hook0(self, input, output):

    global Feature1
    
    features = output.data.cpu().numpy()
    # print("feature size is:", features.shape)
    features = np.squeeze(features)
    # print("feature size is:", features.shape)

    Feature1 += list(features) 

def get_features_hook1(self, input, output):

    global Feature2
    
    features = output.data.cpu().numpy()
    # print("feature size is:", features.shape)
    features = np.squeeze(features)
    # print("feature size is:", features.shape)

    Feature2 += list(features)

def get_features_hook2(self, input, output):

    global Feature3
    
    features = output.data.cpu().numpy()
    # print("feature size is:", features.shape)
    features = np.squeeze(features)
    # print("feature size is:", features.shape)

    Feature3 += list(features)  


def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features, labels = [], []
    total = 0
    global Feature1, Feature2, Feature3
    model.eval()

    
    for index, (img, label) in enumerate(dataloader):

        if total > max_images:
            break

        img, label = img.cuda(), label.cuda()

        # print("model structure is:", model)

        handle0 = model.module.layer1[2].register_forward_hook(get_features_hook0)
        handle1 = model.module.layer2[3].register_forward_hook(get_features_hook1)
        handle2 = model.module.layer3[5].register_forward_hook(get_features_hook2)

        features += list(model(img).data.cpu().numpy())



        handle0.remove()
        handle1.remove()
        handle2.remove()

        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    # print("feature size is:", len(features))
    return np.array(features), np.array(labels)






def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument("--exp-name", type=str, default="temp_eval_ssd")
    parser.add_argument("--results-dir", type=str, default="./eval_results")

    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--classes", type=int, default=10)

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

    results_file = os.path.join(args.results_dir, args.exp_name + "_ssd.txt")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(results_file, "a"))
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    model = SSLResNet(arch=args.arch).eval()
    model.encoder = nn.DataParallel(model.encoder).to(device)

    print("model structure is:", model)

    # load checkpoint
    ckpt_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    model.load_state_dict(ckpt_dict)

    # dataloaders
    train_loader, test_loader, norm_layer = data.__dict__[args.dataset](
        args.data_dir,
        args.batch_size,
        mode=args.data_mode,
        normalize=args.normalize,
        size=args.size,
    )

    # print("model.encoder is:", model.encoder)



    global Feature1, Feature2, Feature3
    Feature1 = []
    Feature2 = []
    Feature3 = []


    # features_train, labels_train = get_features(
    #     model.encoder, train_loader
    # )  # using feature befor MLP-head
    # Feature1_train = np.array(Feature1)
    # Feature2_train = np.array(Feature2)
    # Feature3_train = np.array(Feature3)
    # print("Feature1 size is:", Feature1_train.shape)
    # print("Feature2 size is:", Feature2_train.shape)
    # print("Feature3 size is:", Feature3_train.shape)
    # print("labes are:", labels_train)
    # np.save("./features_all/self_supervised_cifar100/1st_block.npy", Feature1_train)
    # np.save("./features_all/self_supervised_cifar100/2nd_block.npy", Feature2_train)
    # np.save("./features_all/self_supervised_cifar100/3rd_block.npy", Feature3_train)

    # np.save("./features_all/self_supervised_cifar100/features_train2.npy",features_train)
    # np.save("./features_all/self_supervised_cifar100/labels_train2.npy",labels_train)
    

    # features_test, labels_test = get_features(model.encoder, test_loader)
    # np.save("./features_all/self_supervised_cifar100/features_test.npy",features_test)
    
    # Feature1_test = np.array(Feature1)
    # Feature2_test = np.array(Feature2)
    # Feature3_test = np.array(Feature3)
    # print("Feature1 size is:", Feature1_test.shape)
    # print("Feature2 size is:", Feature2_test.shape)
    # print("Feature3 size is:", Feature3_test.shape)
    # print("labels are:", labels_test)
    # np.save("./features_all/self_supervised_cifar100/test_1st_block.npy", Feature1_test)
    # np.save("./features_all/self_supervised_cifar100/test_2nd_block.npy", Feature2_test)
    # np.save("./features_all/self_supervised_cifar100/test_3rd_block.npy", Feature3_test)

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
        features_ood, labels_ood = get_features(model.encoder, ood_loader)
        # np.save("./features_all/self_supervised_cifar100/features_ood.npy",features_ood)
        print("Out-of-distribution features shape: ", features_ood.shape)

        Feature1_ood = np.array(Feature1)
        Feature2_ood = np.array(Feature2)
        Feature3_ood = np.array(Feature3)
        print("Feature1 size is:", Feature1_ood.shape)
        print("Feature2 size is:", Feature2_ood.shape)
        print("Feature3 size is:", Feature3_ood.shape)
        print("labels are:", labels_ood)
        # np.save("./features_all/self_supervised_cifar100/ood_1st_block.npy", Feature1_ood)
        # np.save("./features_all/self_supervised_cifar100/ood_2nd_block.npy", Feature2_ood)
        # np.save("./features_all/self_supervised_cifar100/ood_3rd_block.npy", Feature3_ood)
        np.save("./features_all/self_supervised_cifar100/labels_ood.npy", labels_ood)


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
