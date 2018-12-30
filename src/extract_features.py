import argparse
import os
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np

import data_handler
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Weight imprinting')
parser.add_argument('--data_dir', default='../data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('--exp_dir', default='../log/base_pretrained', type=str,
                    help='experiment directory (default: ../log/base_pretrained')
parser.add_argument('--fts_save_path', default='../data/base_pretrained_features',
                    help='path to save extracted features.')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_data_augments', default=5, type=int,
                    help='number of times that a sample is augmented')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--n_shots', default=1, type=int,
                    help='determines how many samples from each novel class will be in the feature set (default: 1)')
parser.add_argument('--seed', default=11, type=int,
                    help='random seed to select novel samples (default: 11)')
parser.add_argument('--d_emb', default=64, type=int,
                    help='dimension of the embeddings (default: 64)')
parser.add_argument('--cosine_sim', action='store_true',
                    help='whether to normalize features (default: False)')

args = parser.parse_args()

def main():

    # initialize the pretrained model
    model = models.Net(
        n_classes=0,
        d_emb=args.d_emb,
        normalize=args.cosine_sim,
        scale=False).to(args.device)
    model_ckpt_path = os.path.join(args.exp_dir, 'model.ckpt')
    ckpt = utils.load_ckpt(model_ckpt_path, args.device)
    utils.load_ckpt2module(ckpt, model, 'model')

    print ('Loading dataset for n-shots:{} with seed:{} ...'.format(args.n_shots, args.seed))
    train_loader, test_loader = data_handler.get_cubloaders(
        args.data_dir,
        n_shots=args.n_shots,
        seed=args.seed, 
        novel_only=False,
        uniform_sampling=False,
        batch_size=args.batch_size,
        n_workers=args.n_workers)

    print ('Extracting features for training set ...')
    train_fts, train_labels = extract_features(model, train_loader, args.n_data_augments)
    print ('Extracting features for test set ...')
    test_fts, test_labels = extract_features(model, test_loader, 1)

    np.savez(
        args.fts_save_path, 
        train_features=train_fts,
        train_labels=train_labels,
        test_features=test_fts,
        test_labels=test_labels)

def extract_features(model, loader, n_data_augments):
    n_data_augments = max(n_data_augments, 1)
    n_samples = len(loader.dataset) * n_data_augments

    features = np.zeros([n_samples, args.d_emb], dtype=np.float32)
    labels = np.zeros([n_samples], dtype=np.int64)
    six = 0

    model.eval()

    pbar = tqdm(desc='feature extraction', total=n_samples, dynamic_ncols=True)
    with torch.no_grad():
        for _ in range(n_data_augments):
            for input, target in loader:
                bs = input.size(0)

                input = input.to(args.device)
                target = target.to(args.device)

                output = model.extract(input)
                labels[six:six+bs] = target.cpu().numpy()
                features[six:six+bs] = output.cpu().numpy()
                six += bs
                pbar.update(bs)
    pbar.close()

    return features, labels

if __name__ == '__main__':
    main()