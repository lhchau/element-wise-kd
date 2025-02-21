import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


fine_to_coarse = {
    "beaver": "aquatic_mammals", "dolphin": "aquatic_mammals", "otter": "aquatic_mammals",
    "seal": "aquatic_mammals", "whale": "aquatic_mammals",

    "aquarium_fish": "fish", "flatfish": "fish", "ray": "fish", "shark": "fish", "trout": "fish",

    "orchid": "flowers", "poppy": "flowers", "rose": "flowers", "sunflower": "flowers", "tulip": "flowers",

    "bottle": "food_containers", "bowl": "food_containers", "can": "food_containers", "cup": "food_containers", "plate": "food_containers",

    "apple": "fruit_and_vegetables", "mushroom": "fruit_and_vegetables", "orange": "fruit_and_vegetables",
    "pear": "fruit_and_vegetables", "sweet_pepper": "fruit_and_vegetables",

    "clock": "household_electrical_devices", "keyboard": "household_electrical_devices", "lamp": "household_electrical_devices",
    "telephone": "household_electrical_devices", "television": "household_electrical_devices",

    "bed": "household_furniture", "chair": "household_furniture", "couch": "household_furniture",
    "table": "household_furniture", "wardrobe": "household_furniture",

    "bee": "insects", "beetle": "insects", "butterfly": "insects", "caterpillar": "insects", "cockroach": "insects",

    "bear": "large_carnivores", "leopard": "large_carnivores", "lion": "large_carnivores",
    "tiger": "large_carnivores", "wolf": "large_carnivores",

    "bridge": "large_man-made_outdoor_things", "castle": "large_man-made_outdoor_things",
    "house": "large_man-made_outdoor_things", "road": "large_man-made_outdoor_things", "skyscraper": "large_man-made_outdoor_things",

    "cloud": "large_natural_outdoor_scenes", "forest": "large_natural_outdoor_scenes",
    "mountain": "large_natural_outdoor_scenes", "plain": "large_natural_outdoor_scenes", "sea": "large_natural_outdoor_scenes",

    "camel": "large_omnivores_and_herbivores", "cattle": "large_omnivores_and_herbivores",
    "chimpanzee": "large_omnivores_and_herbivores", "elephant": "large_omnivores_and_herbivores", "kangaroo": "large_omnivores_and_herbivores",

    "fox": "medium_mammals", "porcupine": "medium_mammals", "possum": "medium_mammals",
    "raccoon": "medium_mammals", "skunk": "medium_mammals",

    "crab": "non-insect_invertebrates", "lobster": "non-insect_invertebrates", "snail": "non-insect_invertebrates",
    "spider": "non-insect_invertebrates", "worm": "non-insect_invertebrates",

    "baby": "people", "boy": "people", "girl": "people", "man": "people", "woman": "people",

    "crocodile": "reptiles", "dinosaur": "reptiles", "lizard": "reptiles", "snake": "reptiles", "turtle": "reptiles",

    "hamster": "small_mammals", "mouse": "small_mammals", "rabbit": "small_mammals",
    "shrew": "small_mammals", "squirrel": "small_mammals",

    "maple_tree": "trees", "oak_tree": "trees", "palm_tree": "trees", "pine_tree": "trees", "willow_tree": "trees",

    "bicycle": "vehicles_1", "bus": "vehicles_1", "motorcycle": "vehicles_1", "pickup_truck": "vehicles_1", "train": "vehicles_1",

    "lawn_mower": "vehicles_2", "rocket": "vehicles_2", "streetcar": "vehicles_2", "tank": "vehicles_2", "tractor": "vehicles_2"
}


class CIFAR100_SUPERCLASS(torchvision.datasets.CIFAR100):
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.coarse_class_to_idx = {_class: i for i, _class in  enumerate(data['coarse_label_names'])}
        
        self.class_to_superclass, self.superclass_to_class = {}, {}
        for i in range(len(self.classes)):
            coarse_idx = self.coarse_class_to_idx[fine_to_coarse[self.classes[i]]]
            self.class_to_superclass[i] = coarse_idx
            if coarse_idx not in self.superclass_to_class:
                self.superclass_to_class[coarse_idx] = []
            self.superclass_to_class[coarse_idx].append(i)
            
def get_cifar100_super_class(
    batch_size=128,
    num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    data_train = CIFAR100_SUPERCLASS(root='./data', train=True, download=True, transform=transform_train)
    data_test = CIFAR100_SUPERCLASS(root='./data', train=False, download=True, transform=transform_test)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=200, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader, len(data_test.classes)