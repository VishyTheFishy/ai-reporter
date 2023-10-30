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
from BNNBench.data.cifar_c import get_cifar10_c, get_cifar100_c
from BNNBench.data.cifar import get_cifar10, get_cifar100
from BNNBench.data.camelyon17 import get_camelyon17
from BNNBench.data.rxrx1 import get_rxrx1


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--ckpt-files",
    nargs="+",
    type=str,
    required=True,
    help="Checkpoint files in a list",
)
parser.add_argument(
    "--data",
    type=str,
    default="cifar10",
    help="which dataset",
    choices=["cifar10", "cifar10-c", "cifar100", "cifar100-c", "rxrx1", "camelyon17"],
)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4],
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.data == "cifar10":
    _, testloader = get_cifar10()
    testloaders = [testloader]
    n_classes = 10
    n_features = 512

elif args.data == "cifar10-c":
    testloaders = get_cifar10_c("all", args.level)
    n_classes = 10
    n_features = 512

elif args.data == "cifar100":
    _, testloader = get_cifar100()
    testloaders = [testloader]
    n_classes = 100
    n_features = 512

elif args.data == "cifar100-c":
    testloaders = get_cifar100_c("all", args.level)
    n_classes = 100
    n_features = 512

elif args.data == "camelyon17":
    _, _, testloader = get_camelyon17()
    testloaders = [testloader]
    n_classes = 2
elif args.data == "rxrx1":
    _, _, testloader = get_rxrx1()
    testloaders = [testloader]
    n_classes = 1139

# Model
print("==> Building model..")
models = []
for ckpt_file in args.ckpt_files:
    if args.data in ["cifar10", "cifar100", "cifar10-c", "cifar100-c"]:
        model_cls = lambda: VGG("VGG16", n_features, n_classes)
    elif args.data == "rxrx1":

        def model_cls():
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            return model

    elif args.data == "camelyon17":

        def model_cls():
            model = densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, n_classes)
            return model

    net = model_cls()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    checkpoint = torch.load(ckpt_file)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    models.append(net)


def brier_scores(logits, truth):
    prob = F.softmax(logits, dim=-1)
    true_prob = torch.take(prob, truth)
    sum_prob2 = torch.sum(prob * prob, dim=-1)
    return (1.0 - true_prob + sum_prob2) / logits.shape[-1]


def ECE_loss(logits, labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def test():
    all_logits = []
    all_uncertainty = []
    all_truth = []
    for testloader in testloaders:
        with torch.no_grad():
            outputs = []
            for net in models:
                all_pred = []
                all_tgt = []
                for batch_idx, (inputs, targets, *_) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = net(inputs)
                    all_pred.append(preds)
                    all_tgt.append(targets)
                output = torch.cat(all_pred, dim=0)
                target = torch.cat(all_tgt, dim=0)
                outputs.append(output)
        output = torch.stack(outputs, dim=0)
        logits = torch.mean(output, dim=0)
        uncertainty = torch.std(output, dim=0)
        all_logits.append(logits)
        all_uncertainty.append(uncertainty)
        all_truth.append(target)

    logits = torch.cat(all_logits, dim=0)
    uncertainty = torch.cat(all_uncertainty, dim=0)
    truth = torch.cat(all_truth, dim=0)
    print(f"Uncertainty: {torch.mean(uncertainty)}")
    _, predicted = logits.max(1)
    total = truth.size(0)
    correct = predicted.eq(truth).sum().item()
    nll = F.cross_entropy(logits, truth, reduction="mean")
    brier = brier_scores(logits, truth).mean()
    ece = ECE_loss(logits, truth).item()
    print(
        f"Accuracy: {correct / total:.4f}, NLL: {nll:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}"
    )


def test_splited():
    for testloader in testloaders:
        all_logits = []
        all_uncertainty = []
        all_truth = []
        with torch.no_grad():
            outputs = []
            for net in models:
                all_pred = []
                all_tgt = []
                for batch_idx, (inputs, targets, *_) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = net(inputs)
                    all_pred.append(preds)
                    all_tgt.append(targets)
                output = torch.cat(all_pred, dim=0)
                target = torch.cat(all_tgt, dim=0)
                outputs.append(output)
        output = torch.stack(outputs, dim=0)
        logits = torch.mean(output, dim=0)
        uncertainty = torch.std(output, dim=0)
        all_logits.append(logits)
        all_uncertainty.append(uncertainty)
        all_truth.append(target)

        logits = torch.cat(all_logits, dim=0)
        uncertainty = torch.cat(all_uncertainty, dim=0)
        truth = torch.cat(all_truth, dim=0)
        print(f"Uncertainty: {torch.mean(uncertainty)}")
        _, predicted = logits.max(1)
        total = truth.size(0)
        correct = predicted.eq(truth).sum().item()
        nll = F.cross_entropy(logits, truth, reduction="mean")
        brier = brier_scores(logits, truth).mean()
        ece = ECE_loss(logits, truth).item()
        print(
            f"Accuracy: {correct / total:.4f}, NLL: {nll:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}"
        )



test()
#test_splited()