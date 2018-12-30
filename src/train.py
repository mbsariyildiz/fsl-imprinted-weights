import argparse
import os
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import models
import numpy as np

import data_handler
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Classifier Trainer')
parser.add_argument('--data_dir', default='../data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('--exp_dir', default='../log/alljoint', type=str,
                    help='experiment directory (default: ../log/alljoint')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate for base convnet (default: 0.001)')
parser.add_argument('--lr_decay', default=0.94, type=float,
                    help='decay rate of the learning rate (default: 0.94)')
parser.add_argument('--lr_decay_steps', default=4, type=int,
                    help='interval for the learning rate decay (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay for embedding and classifier parameters (default: 1e-4)')
parser.add_argument('--n_shots', default=1, type=int,
                    help='number of novel sample (default: 1)')
parser.add_argument('--seed', default=11, type=int,
                    help='seed set before sampling novel samples for training (default: 11)')
parser.add_argument('--d_emb', default=64, type=int,
                    help='dimension of the embeddings (default: 64)')
parser.add_argument('--cosine_sim', action='store_true',
                    help='whether to compute class scores by computing cosine similarity')
parser.add_argument('--fine_tune', action='store_true',
                    help='whether to fine-tune base convnet')
args = parser.parse_args()

def main():

    utils.prepare_directory(args.exp_dir)
    utils.write_logs(args,
               os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    os.pardir)))

    model = models.Net(
        n_classes=200 if args.n_shots > 0 else 100,
        d_emb=args.d_emb,
        normalize=args.cosine_sim,
        scale=args.cosine_sim).to(args.device)
    utils.model_info(model, 'network', args.exp_dir)

    # Data loading code
    train_loader, test_loader = data_handler.get_cubloaders(
        args.data_dir,
        n_shots=args.n_shots, 
        seed=args.seed, 
        uniform_sampling=args.n_shots>0,
        batch_size=args.batch_size,
        n_workers=args.n_workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.fine_tune:
        print ('Base convnet will be fine-tuned.')
        optimizer = torch.optim.SGD(
            [
                {'params': model.extractor.parameters()},
                {'params': list(model.embedding.parameters()) + list(model.classifier.parameters()) , 
                     'lr': args.lr * 10 }
            ],
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print ('Base convnet will be freezed during training.')
        for p in model.extractor.parameters(): p.requires_grad = False
        optimizer = torch.optim.SGD(
            list(model.embedding.parameters()) + list(model.classifier.parameters()),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_decay, step_size=args.lr_decay_steps)

    best_top1 = 0

    logger_titles = [
        'lr',
        'train_loss',
        'test_loss',
        'train_top1',
        'test_top1',
        'best_top1',
        'avg_recall',
        'base_recall',
        'novel_recall',
        'train_time',
        'test_time']
    logger = utils.Logger(args.exp_dir, 'logs', logger_titles)

    for epoch in trange(args.start_epoch, args.n_epochs, desc='epoch_monitor', dynamic_ncols=True):
        opt_scheduler.step()
        lr = opt_scheduler.get_lr()[-1]

        # train for one epoch
        train_loss, train_top1, train_time = utils.train(model, train_loader, optimizer, criterion, args.device)

        # evaluate on test set
        test_loss, test_top1, confmat, test_time = utils.evaluate(model, test_loader, criterion, args.device)

        pc_rec = np.diag(confmat) / np.sum(confmat, axis=1)
        base_recall = pc_rec[:100].mean() * 100.
        novel_recall = pc_rec[100:].mean() * 100.
        avg_recall = pc_rec.mean() * 100.

        # remember best prec@1 and save checkpoint
        is_best = test_top1 > best_top1
        best_top1 = max(test_top1, best_top1)

        # append logger file
        logger.append(
            [lr, train_loss, test_loss, 
             train_top1, test_top1, best_top1, 
             avg_recall, base_recall, novel_recall,
             train_time, test_time],
            epoch)

        utils.save_checkpoint(
            {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_top1': best_top1,
                'test_top1': test_top1
            },
            args.exp_dir,
            is_best)

    logger.close()

    # save the last confmat
    np.savez(os.path.join(args.exp_dir, 'confmat.npz'), confmat=confmat)

if __name__ == '__main__':
    main()
