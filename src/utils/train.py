from timeit import default_timer as timer
from tqdm import tqdm
from .misc import AverageMeter
from .eval import accuracy

__all__ = ['train']

def train(model, train_loader, optimizer, criterion, device):
    t0 = timer()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    pbar = tqdm(desc='training loop', total=len(train_loader.dataset), dynamic_ncols=True)
    for input, target in train_loader:
        
        input = input.to(device)
        target = target.to(device, non_blocking=True)
        
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(input.size(0))
    pbar.close()
    t1 = timer()

    return (losses.avg, top1.avg, t1 - t0)