from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset, TorchVOCSegmentation,DemoSegmentation,FewShotSegmentation,FewShotSegmentation2
from .transforms import get_transforms
from torchvision.datasets import VOCSegmentation

segment_datasets = {'base_sem': BaseSemanticDataset,'voc_sem': VOCSemanticDataset, 'torch_voc_sem': TorchVOCSegmentation,'demo_sem':DemoSegmentation,'few_sem':FewShotSegmentation,'few_sem2':FewShotSegmentation2}


def get_dataset(cfg):
    name = cfg.name
    assert name in segment_datasets, \
        print('{name} is not supported, please implement it first.'.format(name=name))

    transform = get_transforms(cfg.transforms)
    target_transform = get_transforms(cfg.target_transforms)
    
    return segment_datasets[name](**cfg.params, transform=transform, target_transform=target_transform)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data
