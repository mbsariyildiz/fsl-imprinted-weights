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
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_data_augments', default=200, type=int,
                    help='number of times that a sample is augmented')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--d_emb', default=64, type=int,
                    help='dimension of the embeddings (default: 64)')
parser.add_argument('--cosine_sim', action='store_true',
                    help='whether to normalize features (default: False)')

args = parser.parse_args()

def main():

    # initialize the pretrained model
    model = models.Net(
        n_classes=100,
        d_emb=args.d_emb,
        normalize=args.cosine_sim,
        scale=False).to(args.device)
    model_ckpt_path = os.path.join(args.exp_dir, 'model.ckpt')
    ckpt = utils.load_ckpt(model_ckpt_path, args.device)
    utils.load_ckpt2module(ckpt, model, 'model')

    # create a separate directory for features
    fts_dir = os.path.join(args.exp_dir, 'features')
    utils.prepare_directory(fts_dir, force_delete=True)

    for n_shots in (1, 2, 5, 10, 20):
        for s_ix, seed in enumerate((11, 22, 33, 44, 55)):

            fts_save_path = os.path.join(fts_dir, 'features_n-shots_{:02d}_{:02d}'.format(n_shots, s_ix))

            print ('Loading dataset for n-shots:{} with seed:{} ...'.format(n_shots, seed))
            train_loader, test_loader = data_handler.get_cubloaders(
                args.data_dir,
                n_shots=n_shots,
                seed=seed, 
                novel_only=False,
                uniform_sampling=False,
                batch_size=args.batch_size,
                n_workers=args.n_workers)

            print ('Extracting features for training set ...')
            train_fts, train_labels = extract_features(model, train_loader, args.n_data_augments, args.d_emb)
            print ('Extracting features for test set ...')
            test_fts, test_labels = extract_features(model, test_loader, 1, args.d_emb)

            np.savez(
                fts_save_path, 
                train_features=train_fts,
                train_labels=train_labels,
                test_features=test_fts,
                test_labels=test_labels)

def extract_features(model, loader, n_data_augments, d_emb):
    n_samples = len(loader.dataset) * n_data_augments

    features = np.zeros([n_samples, d_emb], dtype=np.float32)
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