from torch.utils.data import Dataset

class CleanSampler(Dataset):
    def __init__(self, dataset, indices, cleanids=None):
        self.dataset = dataset
        self.indices = indices
        self.cleanids = cleanids

    def __getitem__(self, idx):
        data = self.dataset[self.cleanids[idx]]
        # data = self.dataset[idx]
        clean_label = self.cleanids[idx]
        return data, clean_label

    def __len__(self):
        return len(self.cleanids)