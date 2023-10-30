"""Train CIFAR10 with PyTorch."""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR


import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet121

import os
import argparse

from BNNBench.backbones.resnet import ResNet101
from BNNBench.backbones.vgg import VGG
from BNNBench.utils.cls_utils import progress_bar
from BNNBench.data.cifar import get_cifar100, get_cifar10
from BNNBench.data.camelyon17 import get_camelyon17
from BNNBench.data.rxrx1 import get_rxrx1

from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=1.0e-3, type=float, help="learning rate")
parser.add_argument(
    "--data",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "tiny-imagenet", "camelyon17", "rxrx1"],
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--n-models", type=int, default=20, help="Number of models to train"
)
parser.add_argument(
    "--n-epochs", type=int, default=200, help="Total number of epochs to train"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.data == "cifar10":
    trainloader, testloader = get_cifar10()
    n_classes = 10
    n_features = 512
elif args.data == "cifar100":
    trainloader, testloader = get_cifar100()
    n_classes = 100
    n_features = 512
elif args.data == "camelyon17":
    trainloader, testloader, _ = get_camelyon17()
    n_classes = 2
elif args.data == "rxrx1":
    trainloader, testloader, _ = get_rxrx1()
    n_classes = 1139


def adjust_learning_rate(n_iter, total_iters, n_ensemble):
    """Decay the learning rate based on schedule"""
    iter_per_model = math.ceil(total_iters / n_ensemble)
    phase = ((n_iter - 1) % iter_per_model) / iter_per_model
    if phase == 0:
        save(n_iter // iter_per_model)
    mult =  0.5 * (math.cos(math.pi * phase) + 1.0)
    print(n_iter, mult)
    return mult

# Training
def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, *_) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
        if not step_scheduler_per_epoch:
            # step scheduler at each iteration
            scheduler.step()
    if step_scheduler_per_epoch:
        scheduler.step()


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Testing...")
    with torch.no_grad():
        for batch_idx, (inputs, targets, *_) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )


def save(model_id):
    # Save checkpoint.
    print("Saving..")
    state = {
        "net": net.state_dict(),
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    if args.data == "rxrx1":
        torch.save(
            state,
            f"./checkpoint/snapshot-ensemble/resnet50_{args.data}/ckpt_{model_id}.pth",
        )
    elif args.data in ["cifar10", "cifar100"]:
        torch.save(
            state,
            f"./checkpoint/snapshot-ensemble/vgg16_{args.data}/ckpt_{model_id}.pth",
        )
    elif args.data == "camelyon17":
        torch.save(state, f"./checkpoint/snapshot-ensemble/densenet121_{args.data}/ckpt_{model_id}.pth")
    else:
        raise ValueError(f"Unknown data: {args.data}")


def init():
    # Model
    print("==> Building model..")
    if args.data == "rxrx1":

        def model_cls():
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            return model

    elif args.data in ["cifar10", "cifar100"]:
        model_cls = lambda: VGG("VGG16", n_features, n_classes)
    elif args.data == "camelyon17":

        def model_cls():
            model = densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, n_classes)
            return model
    net = model_cls()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # rxrx1:
    if args.data == "rxrx1":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.data in ["cifar10", "cifar100"]:
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.data == "camelyon17":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=1.0e-2, momentum=0.9)
    scheduler = LambdaLR(
        optimizer,
        lambda n_iter: adjust_learning_rate(
            n_iter, args.n_epochs * len(trainloader), args.n_models
        ),
    )
    step_scheduler_per_epoch = False
    return net, criterion, optimizer, scheduler, step_scheduler_per_epoch


net, criterion, optimizer, scheduler, step_scheduler_per_epoch = init()
epoch_per_model =  args.n_epochs // args.n_models

for epoch in range(args.n_epochs):
    train()
    test()