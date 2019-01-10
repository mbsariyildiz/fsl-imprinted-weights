import argparse
import os
from tqdm import trange, tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import models
import numpy as np

import data_handler
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Weight imprinting + fine tuning')
parser.add_argument('--data_dir', default='../data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('--exp_dir', default='../log/ft-imprinting', type=str,
                    help='experiment directory (default: ../log/ft-imprinting')
parser.add_argument('--model_ckpt_path', default='../log/base-pretrained/model.ckpt', type=str,
                    help='checkpoint file for the model (default: ../log/base-pretrained/model.ckpt')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_epochs', default=200, type=int,
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
parser.add_argument('--random', action='store_true', 
                    help='whether use random novel weights')
parser.add_argument('--n_data_augments', default=10, type=int,
                    help='number of times that a sample is augmented during imprinting')
args = parser.parse_args()

def main():

    utils.prepare_directory(args.exp_dir)
    utils.write_logs(args, None)

    model = models.Net(
        n_classes=100,
        d_emb=args.d_emb,
        normalize=True,
        scale=True).to(args.device)
    utils.model_info(model, 'base-network', args.exp_dir)

    # load pretrained model
    ckpt = utils.load_ckpt(args.model_ckpt_path, args.device)
    utils.load_ckpt2module(ckpt, model, 'model')

    if not args.random:
        print ('Loading training data of novel classes for imprinting ...')
        imprinting_loader, _ = data_handler.get_cubloaders(
            args.data_dir,
            n_shots=args.n_shots,
            seed=args.seed,
            novel_only=True,
            uniform_sampling=False,
            batch_size=args.batch_size,
            n_workers=args.n_workers)

        print ('Imprinting new weights for novel classes using training data of novel classes...')
        utils.imprint(model, imprinting_loader, args.n_data_augments, False, args.d_emb, args.device)

    else:
        print ('Imprinting new random weights for novel classes ...')
        utils.imprint(model, random=True, d_emb=args.d_emb, device=args.device)

    utils.model_info(model, 'imprinted-network', args.exp_dir)
    for p in model.parameters(): p.requires_grad = True

    print ('Loading data for fine-tuning the whole model ...')
    train_loader, test_loader = data_handler.get_cubloaders(
        args.data_dir,
        n_shots=args.n_shots, 
        seed=args.seed,
        novel_only=False,
        uniform_sampling=args.n_shots>0,
        batch_size=args.batch_size,
        n_workers=args.n_workers)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )

    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_decay, step_size=args.lr_decay_steps)

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

    best_top1 = 0

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
        best_top1 = max(test_top1, best_top1)

        ## append logger file
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
        args.exp_dir)

    np.savez(os.path.join(args.exp_dir, 'confmat.npz'), confmat=confmat)

    logger.close()
    print ('')

if __name__ == '__main__':
    main()
