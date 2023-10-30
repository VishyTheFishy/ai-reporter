"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
parser.add_argument("--k-model-start", type=int, default=0, help="Id of first model")
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

# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
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
        if not step_scheduler_per_epoch and scheduler is not None:
            # step scheduler at each iteration
            scheduler.step()
    if step_scheduler_per_epoch and scheduler is not None:
        scheduler.step()


def test(epoch, model_id):
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


def save():
    # Save checkpoint.
    print("Saving..")
    state = {
        "net": net.state_dict(),
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    if args.data == "rxrx1":
        torch.save(state, f"./checkpoint/resnet50_{args.data}/ckpt_{model_id}.pth")
    elif args.data in ["cifar10", "cifar100"]:
        torch.save(state, f"./checkpoint/vgg16_{args.data}/ckpt_{model_id}.pth")
    elif args.data == "camelyon17":
        torch.save(state, f"./checkpoint/densenet121_{args.data}/ckpt_{model_id}.pth")


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

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]

    criterion = nn.CrossEntropyLoss()
    # rxrx1:
    if args.data == "rxrx1":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=args.n_epochs * len(trainloader),
            num_warmup_steps=5415,
        )
        step_scheduler_per_epoch = False
    elif args.data in ["cifar10", "cifar100"]:
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        step_scheduler_per_epoch = True
    elif args.data == "camelyon17":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = None
        step_scheduler_per_epoch = False
    return net, criterion, optimizer, scheduler, step_scheduler_per_epoch


for model_id in range(args.k_model_start, args.k_model_start + args.n_models):
    net, criterion, optimizer, scheduler, step_scheduler_per_epoch = init()
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        train(epoch)
        if testloader is not None:
            test(epoch, model_id)
        save()
