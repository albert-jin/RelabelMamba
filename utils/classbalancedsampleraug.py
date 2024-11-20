from torch.utils.data.sampler import *
import torch

class ClassBalancedSamplerAug(Sampler[int]):

    def __init__(self, labels, num_classes, num_samples=None, num_fold=1, scores=None):
        self.num_fold = num_fold
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.classes = torch.arange(num_classes)
        self.num_classes = torch.as_tensor([torch.sum(self.labels == i) for i in self.classes], dtype=torch.int)
        if num_samples is not None and num_fold is None:
            self.num_fold = torch.floor(torch.tensor(num_samples / len(labels)))
        self.max_num = self.num_classes.max() * self.num_fold
        self.scores = scores
        ids = []
        for i, cid in enumerate(self.classes):
            if self.num_classes[i] == 0:
                continue
            # else:
            #     fold_i = torch.ceil(self.max_num / self.num_classes[i]).to(torch.int)
            # tmp_i = torch.where(self.labels == cid)[0].repeat(fold_i)
            # rand = torch.randperm(self.num_classes.max())
            # tmp_i[-self.num_classes.max():] = tmp_i[-self.num_classes.max():][rand]
            # ids.append(tmp_i[:self.max_num])

            cls_ids = torch.where(self.labels == cid)[0]
            sorted_idx = torch.argsort(self.scores[cls_ids], descending=True)
            cls_ids = cls_ids[sorted_idx]

            tmp_i = cls_ids.repeat(self.num_fold)


            if len(cls_ids) < self.num_classes.max():
                half_count = len(cls_ids) // 2
                if half_count > 0:
                    high_ids = cls_ids[:half_count]
                    tmp_i = torch.cat((tmp_i, high_ids))
            rand = torch.randperm(len(tmp_i))
            ids.append(tmp_i[rand])


        self.ids = torch.cat(ids)

    def __iter__(self):
        rand = torch.randperm(len(self.ids))
        ids = self.ids[rand]
        return iter(ids.tolist())

    def __len__(self):
        return len(self.ids)
