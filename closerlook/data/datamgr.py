# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from .additional_transforms import ImageJitter
from .dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == "RandomSizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Scale":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = [
                "RandomSizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize",
            ]
        else:
            transform_list = ["Scale", "CenterCrop", "ToTensor", "Normalize"]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size):
        super(SimpleDataManager, self).__init__()
        self.trans_loader = TransformLoader(image_size)

    def get_dataset(self, data_file, aug):
        """Return the SimpleDataset corresponding to data_file and aug.

        Parameters
        ----------
        data_file : string
            The json file containing the dataset specifications.

        aug : bool
            If True, using data augmentations.

        Returns
        -------
        data_set : SimpleDataset
            The dataset corresponding to the provided data_file and aug.

        """
        transform = self.trans_loader.get_composed_transform(aug)
        return SimpleDataset(data_file, transform)

    def get_data_loader(
        self, data_file, batch_size, aug
    ):  # parameters that would change on train/val set
        dataset = self.get_dataset(data_file, aug)
        data_loader_params = dict(
            batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

    def get_episode_dataset(self, data_file, episodes, loader=None):
        """Return the EpisodeDataset corresponding to data_file.

        Parameters
        ----------
        data_file : string
            The path to the file list for the desired dataset.
        
        episodes : list
            List of episodes to be included in the dataset.
        
        loader : CachedLoader (optional)
            If not None, use this loader.

        Returns
        -------
        data_set : EpisodeDataset
            The episode dataset.

        """
        if loader is None:
            transform = self.trans_loader.get_composed_transform(aug=False)
            loader = CachedLoader(transform)

        return EpisodeDataset(data_file, episodes, loader)


class CachedLoader:
    def __init__(self, transform):
        self.transform = transform
        self._cache = {}

    def __getitem__(self, file_names):
        if isinstance(file_names, list):
            return self.load_images(file_names)
        elif isinstance(file_names, str):
            return self.load_image(file_names)
        else:
            raise ValueError(
                "loading for type {} not implemented".format(type(file_names))
            )

    def load_image(self, file_name):
        """Load and return the image associated with a file name.

        Parameters
        ----------
        file_name : string
            The file name to be loaded.

        Returns
        -------
        image : torch.Tensor
            Resulting image tensor.

        """
        if file_name not in self._cache:
            self._cache[file_name] = self.transform(
                Image.open(file_name).convert("RGB")
            )

        return self._cache[file_name]

    def load_images(self, file_names):
        """Load images associate with a list of file names.

        Parameters
        ----------
        file_names : list
            List of file names to be loaded.

        Returns
        -------
        images : torch.Tensor
            Resulting image tensor.

        """
        images = []
        for file_name in file_names:
            images.append(self.load_image(file_name).unsqueeze(0))

        return torch.cat(images, 0)


class EpisodeDataset:
    def __init__(self, data_file, episodes, loader):
        """Dataset capturing specified episodes for a dataset.

        Parameters
        ----------
        data_file : string
            Path to the file list.

        episodes : list
            List of episodes in the dataset.

        loader : CachedLoader
            The loader to be used for loading up images.

        """
        with open(data_file, "r") as f:
            self.meta = json.load(f)

        self.episodes = episodes
        self.loader = loader

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, ind):
        episode = self.episodes[ind]

        return {
            "support_inputs": self.loader[episode["support_filenames"]],
            "support_labels": torch.LongTensor(episode["support_labels"]),
            "query_inputs": self.loader[episode["query_filenames"]],
            "query_labels": torch.LongTensor(episode["query_labels"]),
        }


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(
        self, data_file, aug
    ):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(
            batch_sampler=sampler, num_workers=12, pin_memory=True
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
