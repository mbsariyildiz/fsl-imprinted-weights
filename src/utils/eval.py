import numpy as np
import torch
from tqdm import tqdm
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from .misc import AverageMeter

__all__ = ['accuracy', 'evaluate']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model, test_loader, criterion, device):
    t0 = timer()
    losses = AverageMeter()
    top1 = AverageMeter()

    labels = np.zeros([len(test_loader.dataset)], dtype=np.int32)
    preds = np.zeros([len(test_loader.dataset)], dtype=np.int32)
    six = 0

    # switch to evaluate mode
    model.eval()

    pbar = tqdm(desc='test loop', total=len(test_loader.dataset), dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
        
            input = input.to(device)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            labels[six:six+input.size(0)] = target.cpu().numpy()
            preds[six:six+input.size(0)] = output.max(dim=1)[1].cpu().numpy()
            six += input.size(0)

            pbar.update(input.size(0))
        pbar.close()
    t1 = timer()

    confmat = confusion_matrix(labels, preds, labels=np.arange(200))

    return (losses.avg, top1.avg, confmat, t1 - t0)