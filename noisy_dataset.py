import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# Transformations
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

# Downloading/Louding CIFAR10 data
trainset = CIFAR10(root='~/data', train=True, download=True)  # , transform = transform_with_aug)
testset = CIFAR10(root='~/data', train=False, download=True)  # , transform = transform_no_aug)
base_labels = {'airplane': 0, 'automobile': 0, 'bird': 0, 'cat': 0, 'deer': 0, 'dog': 0, 'frog': 0, 'horse': 0, 'ship': 0, 'truck': 0}


class CIFAR10Noisy(Dataset):
    def __init__(self, use_aug=True, n_pairs=5000):
        self.use_aug = use_aug
        self.data = []
        self.targets = []
        stats = base_labels.copy()
        # Arrange target indexes according to classes
        class_index = [np.where(np.array(trainset.targets) == i)[0] for i in range(10)]
        rng = np.random.default_rng()
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
        target_arr = np.fromiter(self.targets[index].values(), dtype=np.int)
        opt_targets = np.where(target_arr == 1)
        target = np.random.choice(opt_targets[0])

        if self.use_aug:
            img = transform_with_aug(img)
        else:
            img = transform_no_aug(img)
        return img, target


class DatasetGenerator:
    def __init__(self, batchSize=128, eval_batch_size=256, dataPath='../../datasets',
                 seed=123, numOfWorkers=4, cutout_length=16):
        self.seed = seed
        np.random.seed(seed)
        self.batchSize = batchSize
        self.eval_batch_size = eval_batch_size
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.cutout_length = cutout_length
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):

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
    ds = CIFAR10Noisy()
    print(ds.__len__())
    ds.__getitem__(500)
