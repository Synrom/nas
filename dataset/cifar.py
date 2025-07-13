import torch
from config import EvalConfig
from torchvision import transforms
from dataset.transform import Cutout


def cifar10_std_and_mean(dataset):
  X = torch.stack([sample[1] for sample in dataset])
  print(X.shape)  # should be 50000, 3, 32, 32
  means = torch.mean(X, dim=(0, 2, 3))
  print(means.shape)  # should be 3
  stds = torch.std(X, dim=(0, 2, 3))
  print(stds.shape)  # should be 3
  return means, stds


def data_transforms_cifar10(config: EvalConfig):
  train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=cifar10_means, std=cifar10_stds),
  ])
  if config.cutout:
    train_transform.transforms.append(Cutout(config.cutout_length))

  eval_transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean=cifar10_means, std=cifar10_stds)])

  return train_transform, eval_transform


def data_transforms_cifar100(config: EvalConfig):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if config.cutout:
    train_transform.transforms.append(Cutout(config.cutout_length))

  valid_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  return train_transform, valid_transform


def data_transforms_cifar(config: EvalConfig):
  if config.cifar100:
    return data_transforms_cifar100(config)
  else:
    return data_transforms_cifar10(config)


cifar10_means = [0.4914, 0.4822, 0.4465]
cifar10_stds = [0.2470, 0.2435, 0.2616]

cifar10_label2name = {
    0: "airplane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

cifar100_label2name = {
    0: "apple (aquatic_mammals)",
    1: "aquarium_fish (aquatic_mammals)",
    2: "baby (aquatic_mammals)",
    3: "bear (aquatic_mammals)",
    4: "beaver (aquatic_mammals)",
    5: "bed (fish)",
    6: "bee (fish)",
    7: "beetle (fish)",
    8: "bicycle (fish)",
    9: "bottle (fish)",
    10: "bowl (flowers)",
    11: "boy (flowers)",
    12: "bridge (flowers)",
    13: "bus (flowers)",
    14: "butterfly (flowers)",
    15: "camel (food_containers)",
    16: "can (food_containers)",
    17: "castle (food_containers)",
    18: "caterpillar (food_containers)",
    19: "cattle (food_containers)",
    20: "chair (fruit_and_vegetables)",
    21: "chimpanzee (fruit_and_vegetables)",
    22: "clock (fruit_and_vegetables)",
    23: "cloud (fruit_and_vegetables)",
    24: "cockroach (fruit_and_vegetables)",
    25: "couch (household_electrical_devices)",
    26: "crab (household_electrical_devices)",
    27: "crocodile (household_electrical_devices)",
    28: "cup (household_electrical_devices)",
    29: "dinosaur (household_electrical_devices)",
    30: "dolphin (household_furniture)",
    31: "elephant (household_furniture)",
    32: "flatfish (household_furniture)",
    33: "forest (household_furniture)",
    34: "fox (household_furniture)",
    35: "girl (insects)",
    36: "hamster (insects)",
    37: "house (insects)",
    38: "kangaroo (insects)",
    39: "keyboard (insects)",
    40: "lamp (large_carnivores)",
    41: "lawn_mower (large_carnivores)",
    42: "leopard (large_carnivores)",
    43: "lion (large_carnivores)",
    44: "lizard (large_carnivores)",
    45: "lobster (large_man-made_outdoor_things)",
    46: "man (large_man-made_outdoor_things)",
    47: "maple_tree (large_man-made_outdoor_things)",
    48: "motorcycle (large_man-made_outdoor_things)",
    49: "mountain (large_man-made_outdoor_things)",
    50: "mouse (large_natural_outdoor_scenes)",
    51: "mushroom (large_natural_outdoor_scenes)",
    52: "oak_tree (large_natural_outdoor_scenes)",
    53: "orange (large_natural_outdoor_scenes)",
    54: "orchid (large_natural_outdoor_scenes)",
    55: "otter (large_omnivores_and_herbivores)",
    56: "palm_tree (large_omnivores_and_herbivores)",
    57: "pear (large_omnivores_and_herbivores)",
    58: "pickup_truck (large_omnivores_and_herbivores)",
    59: "pine_tree (large_omnivores_and_herbivores)",
    60: "plain (medium_mammals)",
    61: "plate (medium_mammals)",
    62: "poppy (medium_mammals)",
    63: "porcupine (medium_mammals)",
    64: "possum (medium_mammals)",
    65: "rabbit (non-insect_invertebrates)",
    66: "raccoon (non-insect_invertebrates)",
    67: "ray (non-insect_invertebrates)",
    68: "road (non-insect_invertebrates)",
    69: "rocket (non-insect_invertebrates)",
    70: "rose (people)",
    71: "sea (people)",
    72: "seal (people)",
    73: "shark (people)",
    74: "shrew (people)",
    75: "skunk (reptiles)",
    76: "skyscraper (reptiles)",
    77: "snail (reptiles)",
    78: "snake (reptiles)",
    79: "spider (reptiles)",
    80: "squirrel (small_mammals)",
    81: "streetcar (small_mammals)",
    82: "sunflower (small_mammals)",
    83: "sweet_pepper (small_mammals)",
    84: "table (small_mammals)",
    85: "tank (trees)",
    86: "telephone (trees)",
    87: "television (trees)",
    88: "tiger (trees)",
    89: "tractor (trees)",
    90: "train (vehicles_1)",
    91: "trout (vehicles_1)",
    92: "tulip (vehicles_1)",
    93: "turtle (vehicles_1)",
    94: "wardrobe (vehicles_1)",
    95: "whale (vehicles_2)",
    96: "willow_tree (vehicles_2)",
    97: "wolf (vehicles_2)",
    98: "woman (vehicles_2)",
    99: "worm (vehicles_2)",
}
