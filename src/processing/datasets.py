import torch
from torch.utils.data import Dataset
import os
import albumentations
import torch
import cv2
import numpy as np
import re


class ImageDatasets(Dataset):
    """Class to generate the dataset to run it in DL models"""
    
    def __init__(self, name, dl_method, disease, augment, path_datasets, path):
        """
        Generate dataset object.
        
        :param name: (str) Name of the dataset used
        :param dl_method: (str) Name of the DL method used
        :param disease: (str) Name of the diseases which are being used to classify with DL
        :param augment (bool) Whether we use the augmented version of the dataset or not
        :param balance (float): Balance of labels between sick and healthy patients for the given disease  TODO Implement
        """

        # initialise some varibles
        self.name = name
        self.dl_method = dl_method
        self.disease = disease
        self.augment = augment

        # get image and label path in a list
        self.path_dataset = path_datasets
        self.path = self.get_path(self.disease, self.name)
        self.folder_path = path

        # get all leaves of the path root
        self.path_walk_image, self.path_walk_label = self.child_node_path(self.path, self.name, self.augment)

        # initialise function for resizing and normalizing the images
        self.IMAGE_SIZE = self.get_image_size(self.dl_method)
        self.aug = albumentations.Compose([
            albumentations.Resize(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], interpolation=1,
                                  always_apply=True),
            albumentations.Normalize(mean=0.445, std=0.269, always_apply=True)
            # grayscale mean and std from ImageNet
        ])

    def __getitem__(self, index):
        """
        Get an item of the dataset object.
        
        :param index: (int) The number we refer to in the dataset
        :return: [image, label] contains the image we refer to with the index inside of the dataset, for example for the 
                index=0, we refer to the image 0.pt of the dataset used and its label
        """

        # generate the image and modify it according to the method used
        image_file_path = self.path_walk_image[index]
        image = torch.load(image_file_path)
        image = (image * 255).astype('uint8') if image.max() <= 1 else image

        if self.dl_method == 'CNN':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.aug(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        # get the label of the image
        label_file_path = self.path_walk_label[index]
        label = torch.tensor(torch.load(label_file_path))

        return [image, label]

    def __len__(self):
        return len(self.path_walk_image)

    def get_path(self, disease, name):
        """
        Gets the right dataset path for the specific disease and dataset name.
        
        :param disease: (str) Either PNEUMONIA or ARDS, which determines if we use the chexpert or mimic dataset
        :param name: (str) The dataset name which was chosen before
        :return: list of paths with length 4
                path_image (str) contains the path to the image folder according to which disease and dataset name was chosen
                path_label (str) contains the path to the label folder according to which disease and dataset name was chosen
                path_image_augmented (str) contains the path to the augmented image folder according to which disease and dataset name was chosen
                path_label_augmented (str) contains the path to the augmented label folder according to which disease and dataset name was chosen
        """
        if disease == 'PNEUMONIA':
            path_image = os.path.join(self.path_dataset, 'chexpert/' + name + '/image')
            path_label = os.path.join(self.path_dataset, 'chexpert/' + name + '/label')
            path_image_augmented = os.path.join(self.path_dataset, 'chexpert_augmented/' + name)
            path_label_augmented = os.path.join(self.path_dataset, 'chexpert_augmented/' + name)
        elif disease == 'ARDS':
            path_image = os.path.join(self.path_dataset, 'mimic/' + name + '/image')
            path_label = os.path.join(self.path_dataset, 'mimic/' + name + '/label')
            path_image_augmented = os.path.join(self.path_dataset, 'mimic_augmented/' + name)
            path_label_augmented = os.path.join(self.path_dataset, 'mimic_augmented/' + name)
        else:
            raise Exception(str("Disease is not supported. Supported diseases are PNEUMONIA AND ARDS."))

        return [path_image, path_label, path_image_augmented, path_label_augmented]

    def child_node_path(self, path, dataset_name, augment):
        """
        Get all child nodes of the path, that means for augmented path, get all augmentation files in each folder for each 
        augmentation technique.
        
        :param path_images: (str) A string path which leads to the images 
        :param path_labels: (str) A string path which leads to the labels
        :param dataset_name: (str) The dataset name of the dataset used
        :param augment: (bool) Whether we use the augmented version of the dataset or not
        :return: path_walk_image (str) contains all image file names which are leaves of the root path
                path_walk_label (str) contains all label file names which are leaves of the root path
        """
        path_walk_image = []
        path_walk_label = []

        with open(os.path.join(self.folder_path, 'aug_tech.txt'), 'r') as f:
            AUG_TECH = [line.strip() for line in f]

        # only save original image paths if is training set or if it is a testset without augmentation
        if dataset_name != 'test' or (dataset_name == 'test' and not augment):
            for file in sorted(os.listdir(path[0]), key=lambda x: float(re.findall("(\d+)", x)[0])):
                path_walk_image.append(path[0] + "/" + file)
                path_walk_label.append(path[1] + "/" + file)

        # only include augmented image path if asked for it
        # for testset only include augmented image path without original/normal testset
        # for trainingset include augmented image path with original/normal trainingset
        if augment:
            for (dir_path, dir_names, file_names) in os.walk(path[2]):
                dir_names.sort()
                if len(file_names) != 0 and os.path.basename(dir_path) == 'image' and \
                        os.path.split(os.path.split(dir_path)[0])[1] in AUG_TECH:
                    for file in sorted(file_names, key=lambda x: float(re.findall("(\d+)", x)[0])):
                        path_walk_image.append(dir_path + "/" + file)
                if len(file_names) != 0 and os.path.basename(dir_path) == 'label' and \
                        os.path.split(os.path.split(dir_path)[0])[1] in AUG_TECH:
                    for file in sorted(file_names, key=lambda x: float(re.findall("(\d+)", x)[0])):
                        path_walk_label.append(dir_path + "/" + file)

        return path_walk_image, path_walk_label

    def get_image_size(self, dl_method):
        """
        Gets the image size according to which deep learning method we are using.
        
        :param dl_method: (str) Either CNN or ViT for the deep learning method which we are using
        :return: (list) The image size accoring to which deep learning method we are using
        """
        if dl_method == 'CNN':
            IMAGE_SIZE = [256, 256]
        elif dl_method == 'VIT':
            IMAGE_SIZE = [224, 224]
        else:
            raise Exception(str("DL method is not supported. Supported DL method are CNN, VIT."))

        return IMAGE_SIZE


class ImageDatasetGenerator:
    def build_dataset(self, dataset_name, dl_method, disease, path, augment=False):
        """
        Function to build the dataset of Class Dataset
        
        :param dataset_name: (str) The name of the dataset used
        :param dl_method: (str) The deep learning method used, either CNN or VIT
        :param disease: (str) Either PNEUMONIA or ARDS, the disease which is being classifies
        :param augment: (bool) Whether the dataset should be taken from the pre augmented or not, default is False
        :return: dataset (Dataset) contains a dataset with the aformentioned properties
        """

        # get available datasets from chexpert and mimic
        datasets_available_pneumonia = [name for name in os.listdir(path + "/chexpert") if
                                        not name.startswith('.') and not name.endswith('.7z')]
        datasets_avaiable_ards = [name for name in os.listdir(path + "/mimic") if
                                  not name.startswith('.') and not name.endswith('.7z')]
        datasets_available = datasets_available_pneumonia + datasets_avaiable_ards

        # raise exception if dataset not available
        if not (dataset_name in datasets_available):
            raise Exception(str("Dataset is not supported. Supported datasets are: " + ', '.join(datasets_available)))

        # generate Dataset Object with properties
        dataset = ImageDatasets(dataset_name, dl_method, disease, augment, path, path)
        return dataset
