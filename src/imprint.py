import argparse
import os
from tqdm import trange, tqdm

import torch
import models
import numpy as np

import data_handler
import utils

parser = argparse.ArgumentParser(description='Weight imprinting')
parser.add_argument('--data_dir', default='../data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('--exp_dir', default='../log/base_pretrained', type=str,
                    help='experiment directory (default: ../log/base_pretrained)')
parser.add_argument('--model_ckpt_path', default='../log/base_pretrained/model.ckpt', type=str,
                    help='path for a ckpt of a pretrained base classifier (default: ../log/base_pretrained/model.ckpt)')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_data_augments', default=200, type=int,
                    help='number of times that a sample is augmented')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--random', action='store_true', 
                    help='whether use random novel weights')
parser.add_argument('--n_shots', default=1, type=int,
                    help='number of novel sample (default: 1)')
parser.add_argument('--seed', default=11, type=int,
                    help='seed set before sampling novel samples for training (default: 11)')
parser.add_argument('--d_emb', default=64, type=int,
                    help='dimension of the embeddings (default: 64)')

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
        utils.imprint(model, imprinting_loader, args.n_data_augments, args.random, args.d_emb, args.device)

    else:
        print ('Imprinting new random weights for novel classes ...')
        utils.imprint(model, random=True, d_emb=args.d_emb, device=args.device)

    utils.model_info(model, 'imprinted-network', args.exp_dir)

    print ('Load test data of all classes for evaluation ...')
    _, test_loader = data_handler.get_cubloaders(
        args.data_dir,
        n_shots=args.n_shots,
        seed=args.seed, 
        novel_only=False,
        uniform_sampling=False,
        batch_size=args.batch_size,
        n_workers=args.n_workers)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    ret = utils.evaluate(model, test_loader, criterion, args.device)

    rec_pc = (np.diag(ret[2]) / np.sum(ret[2], axis=1)) * 100
    base_rec = rec_pc[:100].mean()
    novel_rec = rec_pc[100:].mean()

    np.savetxt(os.path.join(args.exp_dir, 'confmat.txt'), ret[2], fmt='%02d')
    np.savez(os.path.join(args.exp_dir, 'logs.npz'), 
        test_top1=ret[1], test_confmat=ret[2], rec_pc=rec_pc, base_rec=base_rec, novel_rec=novel_rec)

if __name__ == '__main__':
    main()