import torch
import torchvision
from torchvision.datasets import ImageFolder
import sklearn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def load_data(path):
    reshape_size = torchvision.transforms.Resize((32,32))
    data_type = torchvision.transforms.ToTensor()
    normalized_metrics = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return ImageFolder(root = path,transform = torchvision.transforms.Compose([reshape_size, data_type, normalized_metrics]))

def split_data(dataset):
    # The training data is 60% of the full data set. The validation and testing data each comprise 20% of the original data.
    train_d, test_d = train_test_split(dataset, test_size=0.2, random_state=30)
    train_d, validation_d = train_test_split(train_d, test_size=0.25, random_state=30)
    return train_d, validation_d, test_d

def train_dataloarder(dataset):
    return DataLoader(dataset=dataset, num_workers=2, shuffle=True, batch_size=10)

def validation_dataloarder(dataset):
    return DataLoader(dataset=dataset, num_workers=2, shuffle=True, batch_size=10)

def test_dataloarder(dataset):
    return DataLoader(dataset=dataset, num_workers=2, shuffle=True, batch_size=10)


