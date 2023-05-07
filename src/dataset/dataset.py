from torch.utils.data import Dataset


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.target_tensor[idx]

    def __len__(self):
        return self.user_tensor.size(0)


class FeatureDataset(Dataset):
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor

    def __getitem__(self, idx):
        return self.input_tensor[idx]

    def __len__(self):
        return self.input_tensor.size(0)


class TransformDataset:
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])

    def __len__(self):
        return len(self._dataset)


class Item2ItemDataset(Dataset):
    def __init__(self, item1_idx_tensor, item2_idx_tensor, item1_tensor, item2_tensor, similarity_tensor):
        self.item1_idx_tensor = item1_idx_tensor
        self.item2_idx_tensor = item2_idx_tensor
        self.item1_tensor = item1_tensor
        self.item2_tensor = item2_tensor
        self.similarity_tensor = similarity_tensor

    def __getitem__(self, idx):
        return self.item1_idx_tensor[idx], self.item2_idx_tensor[idx],\
               self.item1_tensor[idx], self.item2_tensor[idx], self.similarity_tensor[idx]

    def __len__(self):
        return self.item1_tensor.size(0)