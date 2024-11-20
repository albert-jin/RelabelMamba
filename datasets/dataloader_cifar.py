from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import json
import random
from torchvision.datasets.cifar import *
from typing import Any, Callable, Optional, Tuple
import torch


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def get_asym_cifar100(root_dir):
    super_class = {}
    super_class['aquatic mammals'] = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    super_class['fish'] = ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout']
    super_class['flowers'] = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
    super_class['food containers'] = ['bottle', 'bowl', 'can', 'cup', 'plate']
    super_class['fruit and vegetables'] = ['apple', 'mushroom', 'orange', 'pear', 'sweet pepper']
    super_class['household electrical devices'] = ['clock', 'keyboard', 'lamp', 'telephone', 'television']
    super_class['household furniture'] = ['bed', 'chair', 'couch', 'table', 'wardrobe']
    super_class['insects'] = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
    super_class['large carnivores'] = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    super_class['large man-made outdoor things'] = ['bridge', 'castle', 'house', 'road', 'skyscraper']
    super_class['large natural outdoor scenes'] = ['cloud', 'forest', 'mountain', 'plain', 'sea']
    super_class['large omnivores and herbivores'] = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
    super_class['medium mammals'] = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
    super_class['non-insect invertebrates'] = ['crab', 'lobster', 'snail', 'spider', 'worm']
    super_class['people'] = ['baby', 'boy', 'girl', 'man', 'woman']
    super_class['reptiles'] = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
    super_class['small mammals'] = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    super_class['trees'] = ['maple tree', 'oak tree', 'palm tree', 'pine tree', 'willow tree']
    super_class['vehicles 1'] = ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train']
    super_class['vehicles 2'] = ['lawn mower', 'rocket', 'streetcar', 'tank', 'tractor']

    classes_to_mix = [[] for _ in range(20)]
    with open('{}/meta'.format(root_dir), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        for j, fine in enumerate(entry['fine_label_names']):
            fine = fine.replace('_', ' ')
            for i, coarse in enumerate(entry['coarse_label_names']):
                coarse = coarse.replace('_', ' ')
                if fine in super_class[coarse]:
                    classes_to_mix[i].append(j)

    return classes_to_mix

class cifar_dataset(Dataset):
    def __init__(self, dataset, noisy_dataset, root_dir, noise_data_dir, transform,  noise_mode='sym',
                 dataset_mode='train', noise_ratio=0.5, open_ratio=0.5, noise_file=None):

        self.r = noise_ratio  # total noise ratio
        self.on = open_ratio  # proportion of open noise
        self.transform = transform
        self.mode = dataset_mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
        self.open_noise = None
        self.closed_noise = None

        if self.mode == 'test':
            if dataset == 'cifar10':
                cifar_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.cifar_data = cifar_dic['data']
                self.cifar_data = self.cifar_data.reshape((10000, 3, 32, 32))
                self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))
                self.cifar_label = cifar_dic['labels']
            elif dataset == 'cifar100':
                cifar_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.cifar_data = cifar_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.cifar_label = cifar_dic['fine_labels']

        elif self.mode == 'train':
            if dataset == 'cifar10':
                cifar_data = []
                cifar_label = []
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    cifar_data.append(data_dic['data'])
                    cifar_label = cifar_label + data_dic['labels']
                self.cifar_data = np.concatenate(cifar_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

            elif dataset == 'cifar100':
                cifar_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                cifar_label = cifar_dic['fine_labels']
                self.cifar_data = cifar_dic['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.clean_label = cifar_label
            if noisy_dataset == 'imagenet32':
                noise_data = None
            else:
                noise_data = unpickle('%s/cifar-100-python/train' % noise_data_dir)['data'].reshape((50000, 3, 32, 32)).transpose(
                    (0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise = json.load(open(noise_file, "r"))
                noise_labels = noise['noise_labels']
                self.open_noise = noise['open_noise']
                # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None
                self.closed_noise = noise['closed_noise']
                for cleanIdx, noisyIdx in noise['open_noise']:
                    if noisy_dataset == 'imagenet32':
                        self.cifar_data[cleanIdx] = np.asarray(
                            Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
                        # set the groundtruth outliers label to be -1
                        self.clean_label[cleanIdx] = 10
                    else:
                        self.cifar_data[cleanIdx] = noise_data[noisyIdx]
                        self.clean_label[cleanIdx] = 10
                self.cifar_label = noise_labels
            else:
                # inject noise
                noise_labels = []  # all labels (some noisy, some clean)
                idx = list(range(50000))  # indices of cifar dataset
                random.shuffle(idx)
                num_total_noise = int(self.r * 50000)  # total amount of noise
                num_open_noise = int(self.on * num_total_noise)  # total amount of noisy/openset images
                print('Statistics of synthetic noisy CIFAR dataset: ', 'num of clean samples: ', 50000 - num_total_noise,
                      ' num of closed-set noise: ', num_total_noise - num_open_noise, ' num of open-set noise: ', num_open_noise)
                # print(num_open_noise, num_total_noise)
                if noisy_dataset == 'imagenet32':  # indices of openset source images
                    target_noise_idx = list(range(1281149))
                else:
                    target_noise_idx = list(range(50000))
                random.shuffle(target_noise_idx)
                self.open_noise = list(
                    zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
                self.closed_noise = idx[num_open_noise:num_total_noise]  # closed set noise indices
                # populate noise_labels
                for i in range(50000):
                    if i in self.closed_noise:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                        elif noise_mode == 'asym':
                            if dataset == 'cifar10':
                                noiselabel = self.transition[cifar_label[i]]
                            elif dataset == 'cifar100':
                                dir_meta = '%s/cifar-100-python' % root_dir
                                transition_100 = get_asym_cifar100(dir_meta)
                                z = [x.copy() for x in transition_100 if cifar_label[i] in x][0]
                                z.remove(cifar_label[i])
                                noiselabel = random.choice(z)

                        noise_labels.append(noiselabel)
                    else:
                        noise_labels.append(cifar_label[i])
                # populate openset noise images
                for cleanIdx, noisyIdx in self.open_noise:
                    if noisy_dataset == 'imagenet32':
                        self.cifar_data[cleanIdx] = np.asarray(
                            Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
                        self.clean_label[cleanIdx] = 10000

                    else:
                        self.cifar_data[cleanIdx] = noise_data[noisyIdx]
                        self.clean_label[cleanIdx] = 10000

                # write noise to a file, to re-use
                noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
                print("save noise to %s ..." % noise_file)
                json.dump(noise, open(noise_file, "w"))
                self.cifar_label = noise_labels
                # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.dataset_mode}!')

    def update_labels(self, new_label):
        self.cifar_label = new_label.cpu()

    def __getitem__(self, index):
        # print(index)
        if self.mode == 'train':
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            clean_target = self.clean_label[index]
            return img, target, clean_target, index
        else:
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            return img, target, index

    def __len__(self):
        return len(self.cifar_data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)

    def __repr__(self):
        return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'


# class cifar_dataset_fixmatch(cifar_dataset):
#     '''
#     Four different dataset modes:
#         all: all train dataset with noisy labels
#         subset: part of train dataset selected by subset_id with noisy labels
#         test: all test dataset with clean labels
#         memory: all train dataset with clean labels
#     '''
#
#     def __init__(self, dataset, root_dir, strong_transform, weak_transform, dataset_mode='train', noise_ratio=0.5, noise_mode='sym',
#                  noise_file=None):
#         '''
#         :param dataset: cifar10 or cifar100
#         :param root_dir: dataset directory
#         :param transform: transform made to images
#         :param dataset_mode: 'train' or 'test' or 'eval'
#         :param noise_ratio: generated sample noise ratio
#         :param noise_mode: 'sym' or 'asym' or 'mixed'
#         :param noise_file: generated noise file path
#         '''
#         # airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
#         super(cifar_dataset_fixmatch, self).__init__(dataset, root_dir, None, dataset_mode, noise_ratio, noise_mode, noise_file)
#         self.strong_transform = strong_transform
#         self.weak_transform = weak_transform
#
#     def __getitem__(self, index):
#         img, target, index = self.cifar_data[index]
#         strong_img = self.strong_transform(img)
#         weak_img = self.weak_transform(img)
#         return weak_img, strong_img, target, index
#
#     def __len__(self):
#         return len(self.cifar_data)

# if __name__ == '__main__':
#     for mode in ['train', 'memory', 'test']:
#         testdata = cifar_dataset(dataset='cifar10',
#                                  root_dir='/Users/chen/Desktop/SSL_noise_reduction/dataset/cifar10',
#                                  dataset_mode=mode, transform=transforms.ToTensor(),
#                                  noise_file='/Users/chen/Desktop/SSL_noise_reduction/cifar10_noise_file.json')
#
#         print(len(testdata))
#         print(testdata[0][1:])
