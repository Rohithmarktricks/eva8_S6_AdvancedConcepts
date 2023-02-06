import torchvision
from typing import Tuple, Any


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    '''Custom class that creates the transformed CIFAR-10 DataSet'''

    def __init__(self, root='~/data/cifar10', train=True, download=True, transform=None):
        super().__init__(root=root, 
                         train=train,
                         download=download,
                         transform=transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label