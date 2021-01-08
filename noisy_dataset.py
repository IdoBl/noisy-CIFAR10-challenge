"""Implementation of noisy CIFAR10 dataset and the relevant DataLoaders
"""


import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Transformations to be used by data loader
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

# CIFAR10 data
trainset = CIFAR10(root='~/data', train=True, download=True)  # , transform = transform_with_aug)
testset = CIFAR10(root='~/data', train=False, download=True)  # , transform = transform_no_aug)
base_labels = {'airplane': 0, 'automobile': 0, 'bird': 0, 'cat': 0, 'deer': 0, 'dog': 0, 'frog': 0, 'horse': 0, 'ship': 0, 'truck': 0}


class CIFAR10Noisy(Dataset):
    """Dataset class which creates the pairs of the images and the label dict as instructed in the challenge"""

    def __init__(self, use_aug=True, n_pairs=5000):
        self.use_aug = use_aug
        self.data = []
        self.targets = []
        stats = base_labels.copy()
        # Arrange target indexes according to classes
        class_index = [np.where(np.array(trainset.targets) == i)[0] for i in range(10)]
        rng = np.random.default_rng()
        # Create trainset image pairs and labels dict
        for _ in range(n_pairs):
            pair_cls = rng.choice(trainset.classes, size=2, replace=False)
            pair_idx = [trainset.class_to_idx[key] for key in pair_cls]
            sample1_idx = rng.choice(class_index[pair_idx[0]])
            sample2_idx = rng.choice(class_index[pair_idx[1]])
            pair_images = (trainset.data[sample1_idx], trainset.data[sample2_idx])
            pair_labels = base_labels.copy()
            pair_labels[pair_cls[1]] = pair_labels[pair_cls[0]] = 1
            self.data.append(pair_images)
            self.targets.append(pair_labels)
            stats[pair_cls[0]] += 1
            stats[pair_cls[1]] += 1
        print(f'Sample spread : {stats}')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_rand_idx = np.random.randint(0, 2)
        img = self.data[index][img_rand_idx]
        target_arr = np.fromiter(self.targets[index].values(), dtype=np.float)
        # No weights to labels yet or same weights
        if 1 in target_arr:
            opt_targets = np.where(target_arr == 1)
            target = np.random.choice(opt_targets[0])
        else:
            target_val = np.random.choice(target_arr, p=target_arr)
            target = np.where(target_arr == target_val)[0][0]

        if self.use_aug:
            img = transform_with_aug(img)
        else:
            img = transform_no_aug(img)
        return img, target, index


class DatasetGenerator:
    """Generator which responsible for loading the batches during the train process"""

    def __init__(self, batch_size=128, eval_batch_size=256, data_path='../../datasets',
                 seed=123, num_of_workers=4, cutout_length=16):
        self.seed = seed
        np.random.seed(seed)
        self.batchSize = batch_size
        self.eval_batch_size = eval_batch_size
        self.dataPath = data_path
        self.numOfWorkers = num_of_workers
        self.cutout_length = cutout_length
        self.data_loaders = self.load_data()
        return

    def get_loader(self):
        return self.data_loaders

    def load_data(self):

        train_dataset = CIFAR10Noisy()

        test_dataset = CIFAR10(root=self.dataPath, train=False,
                               transform=transform_no_aug, download=True)

        data_loaders = {'train_dataset': DataLoader(dataset=train_dataset,
                                                    batch_size=self.batchSize,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=self.numOfWorkers),
                        'test_dataset': DataLoader(dataset=test_dataset,
                                                   batch_size=self.eval_batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)}

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders


if __name__ == '__main__':
    data = [[-0.08593865,  0.54177475,  0.08312026, -0.1020374,  -0.4152222,  -0.25962165,0.1514074,   0.40518272, -0.7945103,   0.41773733],
 [-0.32546175,  0.40577203, -0.2636229,  -0.5520464,   0.55566704, -0.07684339,
  -0.30999124, -0.19996691,  0.14118223,  0.64797604],
 [ 0.36379483,  0.27278993, -0.449432,    0.27793324, -0.48690692,  0.10115878,
   0.18801644, -0.15390219,  0.27907926, -0.1649591 ],
 [-0.6337068,  -0.5081649,   0.21615173,  0.01803527, -0.1222709,  -0.22864321,
   0.12365295,  0.5406585,  -0.5175655,   0.2193884 ]]
    ds = CIFAR10Noisy()
    pred_arr = np.array(data, dtype=np.float)
    indexes_arr = np.array([1649,  182, 4212, 3895], dtype=np.int)
    threshold = 0.3
    highest_pred = np.argmax(pred_arr, axis=1)
    for idx, high_pred in enumerate(highest_pred):
        if np.where(pred_arr[idx][high_pred] >= threshold, True, False) and list(ds.targets[indexes_arr[idx]].values())[high_pred] != 0:
            key = list(ds.targets[indexes_arr[idx]].keys())[high_pred]
            labels_idx = np.where(np.array(list(ds.targets[indexes_arr[idx]].values())) != 0)[0]
            if labels_idx[0] == high_pred:
                other_key = list(ds.targets[indexes_arr[idx]].keys())[labels_idx[1]]
                ds.targets[indexes_arr[idx]][key] *= 1.02
                ds.targets[indexes_arr[idx]][other_key] *= 0.98
            else:
                other_key = list(ds.targets[indexes_arr[idx]].keys())[labels_idx[0]]
                ds.targets[indexes_arr[idx]][key] *= 1.02
                ds.targets[indexes_arr[idx]][other_key] *= 0.98


    print(ds.__len__())
    ds.__getitem__(500)
