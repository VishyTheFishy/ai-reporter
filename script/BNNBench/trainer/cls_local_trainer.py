'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet121

import copy
import os
import argparse

from BNNBench.backbones.resnet import ResNet101
from BNNBench.backbones.vgg import VGG
from BNNBench.utils.cls_utils import progress_bar
from BNNBench.data.cifar import get_cifar10, get_cifar100
from BNNBench.data.camelyon17 import get_camelyon17
from BNNBench.data.rxrx1 import get_rxrx1

from transformers import get_cosine_schedule_with_warmup


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'tiny-imagenet', 'camelyon17', 'rxrx1'])
parser.add_argument('--lr', default=1.0e-3, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.data == 'cifar10':
    trainloader, testloader = get_cifar10()
    n_classes = 10
    n_features = 512
elif args.data == 'cifar100':
    trainloader, testloader = get_cifar100()
    n_classes = 100
    n_features = 512
elif args.data == "camelyon17":
    trainloader, testloader, _ = get_camelyon17()
    n_classes = 2
elif args.data == "rxrx1":
    trainloader, testloader, _ = get_rxrx1()
    n_classes = 1139

def dist_to_anchor(model):
    d = 0.0
    n = 0
    for net_id, net_anchor in enumerate(net_anchors):
        di = 0
        ni = 0
        for w, w_anchor in zip(model.parameters(), net_anchor.parameters()):
            di += F.l1_loss(w, w_anchor.detach(), reduction="sum")
            ni += w.numel()
        d += di
        n += ni
    return d / n

# Training
def train(n_iters, lam):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    k_iter = 0
    while k_iter < n_iters:
        for batch_idx, (inputs, targets, *_) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            anchor_loss = dist_to_anchor(net)
            ce_loss = criterion(outputs, targets)
            loss = ce_loss - lam * anchor_loss        
            loss.backward()
            optimizer.step()
            if not step_scheduler_per_epoch and scheduler is not None:
                scheduler.step()
            train_loss += ce_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Anchor dist: %.6f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), anchor_loss.item(), 100.*correct/total, correct, total))
            k_iter += 1
            if k_iter == n_iters:
                break
        if step_scheduler_per_epoch and scheduler is not None:
            scheduler.step()

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

def test(epoch):
    correct = 0
    nll = 0
    brier = 0
    total = 0
    ece = 0
    with torch.no_grad():
        outputs = []
        for net in models:
            net.eval()
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
    print(f"Uncertainty: {torch.mean(uncertainty)}")
    _, predicted = logits.max(1)
    total = target.size(0)
    correct = predicted.eq(target).sum().item()
    nll = F.cross_entropy(logits, target, reduction="mean")
    brier = brier_scores(logits, target).mean()
    ece = ECE_loss(logits, target).item()
    print(f"Accuracy: {correct / total:.4f}, NLL: {nll:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}")
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    if args.data == "rxrx1":
        torch.save(state, f'./checkpoint/resnet50_{args.data}/local3_{model_id}_{epoch}.pth')
    elif args.data in ["cifar10", "cifar100"]:
        torch.save(state, f'./checkpoint/vgg16_{args.data}/local0_{model_id}_{epoch}.pth')
    elif args.data == "camelyon17":
        torch.save(state, f'./checkpoint/densenet121_{args.data}/local0_{model_id}_{epoch}.pth')

def init(model_in=None):
    # Model
    if model_in is None:
        print('==> Building model..')
        if args.data in ["cifar10", "cifar100"]:
            model_cls = lambda: VGG("VGG16", n_features, n_classes)
            checkpoint = torch.load(f'./checkpoint/vgg16_{args.data}/ckpt_0.pth')
        elif args.data == "rxrx1":
            def model_cls():
                model = resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, n_classes)
                return model
            checkpoint = torch.load(f'./checkpoint/resnet50_{args.data}/ckpt_3.pth')
        elif args.data == "camelyon17":
            def model_cls():
                model = densenet121(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, n_classes)
                return model
            checkpoint = torch.load(f'./checkpoint/densenet121_{args.data}/ckpt_0.pth')

        net = model_cls()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        net_anchor = model_cls().to(device)
        net_anchor = torch.nn.DataParallel(net_anchor)
        net_anchor.load_state_dict(checkpoint['net'])
        net.load_state_dict(checkpoint['net'])
    else:
        net = copy.deepcopy(model_in)
        net_anchor = copy.deepcopy(model_in)
    criterion = nn.CrossEntropyLoss()
    if args.data == "rxrx1":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=10 * len(trainloader),
                num_warmup_steps=256)
        step_scheduler_per_epoch = False
    elif args.data in ["cifar10", "cifar100"]:
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
        def lr_fn(epoch):
            if epoch < 5:
                return 1
            elif epoch < 9:
                return 0.1
            elif epoch < 10:
                return 0.01
            else:
                return 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
        step_scheduler_per_epoch = True
    elif args.data == "camelyon17":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.01, momentum=0.)
        scheduler = None
        step_scheduler_per_epoch = False

    return net_anchor, net, criterion, optimizer, scheduler, step_scheduler_per_epoch

net_anchors = []
models = []
model_in = None
for model_id in range(6):
    net_anchor, net, criterion, optimizer, scheduler, step_scheduler_per_epoch = init(model_in)
    if model_id == 0:
        net_anchors = [net_anchor]
    """
    if model_id == 0:
        net_anchors.append(net_anchor)
    """
    train(int(3 * len(trainloader)), 5.0)
    print("Finetuning...")
    models.append(net)
    train(int(3 * len(trainloader)), 0.0)
    test(1)
    net_anchors = [net]
    model_in = net

"""
Camelyon17  0.25   0.25   1500.0
rxrx1       3      3      15
cifar10     3      3      5
cifar100
"""
    