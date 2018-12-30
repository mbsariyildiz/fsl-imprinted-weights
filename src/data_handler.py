import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

join = os.path.join
np.set_printoptions(linewidth=150, precision=3, suppress=True)

cub_helper = None
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

class CUBHelper(object):

    def __init__(self, data_dir, val_split=0.0, n_shots=0, seed=99, novel_only=False, normalize_atts=True):

        self.data_dir = data_dir
        self.image_dir = join(self.data_dir, 'images')
        self.val_split = val_split
        self.n_shots = n_shots
        self.novel_only = novel_only
        self.normalize_atts = normalize_atts
        
        print ('**** CUB-200-2011 Helper ****')
        _order = np.arange(200)
        self.base_class_inds = _order[:100]
        self.novel_class_inds = _order[100:]
    
        attributes = []
        atts_file = join(self.data_dir, 'attributes',  'class_attribute_labels_continuous.txt')
        with open(atts_file, 'r') as fid_atts:
            for line in fid_atts:
                atts = line.strip().split()
                atts = list(map(float, atts))
                attributes.append(atts)
        self.attributes = np.asarray(attributes).astype(np.float32) / 100.
        assert np.all(np.isfinite(self.attributes))
        if self.normalize_atts:
            print ('Normalizing dataset attributes ... ')
            self.attributes = self.attributes / np.linalg.norm(self.attributes, axis=1, keepdims=True)
        
        triplets = {}
        fid_paths = open(join(self.data_dir, 'images.txt'), 'r')
        fid_splits = open(join(self.data_dir, 'train_test_split.txt'), 'r')
        fid_labels = open(join(self.data_dir, 'image_class_labels.txt'), 'r')

        for path_line, split_line, label_line in zip(fid_paths, fid_splits, fid_labels):
            sample_id, rel_path = path_line.strip().split()
            _, split = split_line.strip().split()
            _, label = label_line.strip().split()
            triplets[sample_id] = [rel_path, label, split]
          
        fid_paths.close()
        fid_splits.close()
        fid_labels.close()
        
        train_image_paths, train_labels = [], []
        test_image_paths, test_labels = [], []
        
        for path, label, split in triplets.values():
            if split == '1':
                train_image_paths.append(join(self.image_dir, path))
                train_labels.append(int(label))
            elif split == '0':
                test_image_paths.append(join(self.image_dir, path))
                test_labels.append(int(label))
            else: raise ValueError('Unrecognized split digit: ', split)
          
        train_image_paths = np.asarray(train_image_paths)
        test_image_paths = np.asarray(test_image_paths)
        train_labels = np.asarray(train_labels) - 1
        test_labels = np.asarray(test_labels) - 1
        
        if val_split > 0.0:
            np.random.seed(67)

            print ('validation split:', val_split)
            n_train = int(train_labels.shape[0] * (1.0 - self.val_split))
            order = np.random.permutation(train_labels.shape[0])

            test_image_paths = train_image_paths[order[n_train:]] 
            test_labels = train_labels[order[n_train:]]

            train_image_paths = train_image_paths[order[:n_train]]
            train_labels = train_labels[order[:n_train]]
          
        assert train_image_paths.shape[0] == train_labels.shape[0]
        assert test_image_paths.shape[0] == test_labels.shape[0]
        assert np.unique(train_labels).shape[0] == 200
        assert np.unique(test_labels).shape[0] == 200
        print ('{} training and ' \
               '{} test images remaining after '\
               '{} validation split.'.format(train_image_paths.shape[0], test_image_paths.shape[0], val_split))

        if self.novel_only:
            # keep only the samples of novel classes
            
            assert self.n_shots > 0

            np.random.seed(seed)
            novel_train_inds = []
            per_class_novel_train_inds = [np.where(train_labels == c)[0] for c in self.novel_class_inds]
            for pc_inds in per_class_novel_train_inds:
                if len(pc_inds) < self.n_shots:
                    novel_train_inds.append(pc_inds)
                else:
                    novel_train_inds.append(np.random.choice(pc_inds, self.n_shots, replace=False))
            novel_train_inds = np.concatenate(novel_train_inds)
            # print ('novel_train_inds:', novel_train_inds)
            # print ('novel_train_inds.shape:', novel_train_inds.shape)
            novel_test_inds = np.concatenate([ np.where(test_labels == c)[0] for c in self.novel_class_inds ])

            self.train_image_paths = train_image_paths[novel_train_inds]
            self.train_labels = train_labels[novel_train_inds]
            self.test_image_paths = test_image_paths[novel_test_inds]
            self.test_labels = test_labels[novel_test_inds]

        else:

            base_train_inds = np.concatenate([ np.where(train_labels == c)[0] for c in self.base_class_inds ])
            base_test_inds = np.concatenate([ np.where(test_labels == c)[0] for c in self.base_class_inds ])

            if self.n_shots > 0:
                # combine D_base with D_novel to construct training and test sets

                np.random.seed(seed)
                novel_train_inds = []
                per_class_novel_train_inds = [np.where(train_labels == c)[0] for c in self.novel_class_inds]
                for pc_inds in per_class_novel_train_inds:
                    if len(pc_inds) < self.n_shots:
                        novel_train_inds.append(pc_inds)
                    else:
                        novel_train_inds.append(np.random.choice(pc_inds, self.n_shots, replace=False))
                novel_train_inds = np.concatenate(novel_train_inds)
                # print ('novel_train_inds:', novel_train_inds)
                # print ('novel_train_inds.shape:', novel_train_inds.shape)

                self.train_image_paths = np.concatenate([
                    train_image_paths[base_train_inds], train_image_paths[novel_train_inds] ])
                self.train_labels = np.concatenate([
                    train_labels[base_train_inds], train_labels[novel_train_inds] ])
                assert self.train_image_paths.shape[0] == self.train_labels.shape[0]

                # keep the test set as it is
                self.test_image_paths = test_image_paths
                self.test_labels = test_labels

            else:
                # keep only the samples of base classes
                self.train_image_paths = train_image_paths[base_train_inds]
                self.train_labels = train_labels[base_train_inds]
                self.test_image_paths = test_image_paths[base_test_inds]
                self.test_labels = test_labels[base_test_inds]

        train_classes = np.unique(self.train_labels)
        test_classes = np.unique(self.test_labels)
        assert np.all(train_classes == test_classes)
        self.n_classes = train_classes.shape[0]

        print ('Helper loaded: ')
        print ('\t{} classes, '.format(self.n_classes))
        print ('\t{} attributes, '.format(self.attributes.shape[1]))
        print ('\t{} training pairs, '.format(self.train_image_paths.shape[0]))
        print ('\t{} test pairs, '.format(self.test_image_paths.shape[0]))
    
class PathLabelDataset(Dataset):

    def __init__(self, image_paths, labels,
                       compute_weights=False,
                       image_transform=transforms.ToTensor()):
        super().__init__()
        
        self.image_paths = image_paths
        self.labels = labels
        self.image_transform = image_transform
        assert self.image_paths.shape[0] == self.labels.shape[0]

        if compute_weights:
            n_samples_per_class = np.array([len(np.where(self.labels == L)[0]) for L in np.unique(self.labels)])
            class_weights = 1. / n_samples_per_class
            sample_weights = np.array([class_weights[L] for L in self.labels])
            self.sample_weights = torch.from_numpy(sample_weights).float()

    def __len__(self):
        return self.image_paths.shape[0]
      
    def __getitem__(self, ix):
        path, label = self.image_paths[ix], self.labels[ix]
        image = Image.open(path).convert('RGB')
        image = self.image_transform(image)
        label = int(label)
        return image, label

def get_cubloaders(data_dir, 
                   val_split=0.0,
                   n_shots=0,
                   seed=99,
                   novel_only=False,
                   batch_size=64,
                   n_workers=4,
                   uniform_sampling=True ):
    
    global cub_helper
    cub_helper = CUBHelper(data_dir, val_split=val_split, n_shots=n_shots, seed=seed, novel_only=novel_only)

    trainloader = _get_cubtrainloader(batch_size, n_workers, uniform_sampling)
    testloader = _get_cubtestloader(batch_size, n_workers)

    return trainloader, testloader

def _get_cubtrainloader(batch_size=64, n_workers=4, uniform_sampling=True):
    assert cub_helper is not None, 'Initialize cub_helper first.'

    trainset = PathLabelDataset(
        cub_helper.train_image_paths, cub_helper.train_labels, 
        compute_weights=uniform_sampling,
        image_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    _sampler = None
    if uniform_sampling: 
        _sampler = WeightedRandomSampler(
            trainset.sample_weights, len(trainset.sample_weights), replacement=True)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size, 
        shuffle=not uniform_sampling,
        sampler=_sampler,
        num_workers=n_workers, 
        pin_memory=True)

    return trainloader

def _get_cubtestloader(batch_size=64, n_workers=4):
    assert cub_helper is not None, 'Initialize cub_helper first.'

    testset = PathLabelDataset(
        cub_helper.test_image_paths, cub_helper.test_labels, 
        compute_weights=False,
        image_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    testloader = DataLoader(
        testset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=n_workers, 
        pin_memory=True)

    return testloader

