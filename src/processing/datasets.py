import os
from processing.datasets_metadata import TimeseriesMetaData, TimeSeriesMetaDataManagement, ImageMetaData
import torch 
import albumentations
from torch.utils.data import Dataset
from pydantic import BaseModel
import pandas as pd
import cv2
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

# ████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗
# ╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝
#    ██║   ██║██╔████╔██║█████╗█████╗███████╗█████╗  ██████╔╝██║█████╗  ███████╗
#    ██║   ██║██║╚██╔╝██║██╔══╝╚════╝╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║
#    ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║
#    ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
class TimeSeriesDataset(BaseModel):
    """
    Represents a timeseries dataset with its associated metadata.
    Contains the actual data content and all metadata about processing steps and dataset characteristics.
    """
    class Config:
        arbitrary_types_allowed = True

    content: pd.DataFrame
    """DataFrame containing the actual timeseries data"""
    
    meta_data: TimeseriesMetaData
    """Metadata object describing dataset characteristics and processing history"""


class TimeSeriesDatasetManagement:
    """
    Manages creation, persistence, and metadata handling for timeseries datasets.
    Handles both creation of new datasets and merging with existing metadata.
    """
    
    @staticmethod
    def factory_method(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                      additional_information: str = None, existing_meta_data: TimeseriesMetaData = None) -> TimeSeriesDataset:
        """
        Factory method to create TimeSeriesDataset instances with appropriate metadata.
        Routes to either existing metadata merge or new metadata creation.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information to include in metadata
            existing_meta_data: Optional existing metadata to merge with new metadata
            
        Returns:
            TimeSeriesDataset object with data and associated metadata
        """
        logger.info(f"Creating TimeSeriesDataset (type: {dataset_type})...")
        logger.debug(f"Dataset shape: {dataset.shape}, Path: {path}")
        
        if existing_meta_data:
            logger.info(f"Creating dataset with merged metadata for {dataset_type} set")
            return TimeSeriesDatasetManagement._factory_existing_meta_data(
                dataset, processing_meta_data, path, dataset_type, additional_information, existing_meta_data
            )
        else:
            logger.info(f"Creating dataset with new metadata for {dataset_type} set")
            return TimeSeriesDatasetManagement._factory_new_meta_data(
                dataset, processing_meta_data, path, dataset_type, additional_information
            )

    @staticmethod
    def _factory_existing_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                                   additional_information: str = None, existing_meta_data: TimeseriesMetaData = None) -> TimeSeriesDataset:
        """
        Creates a TimeSeriesDataset by merging new metadata with existing metadata.
        Used when processing steps have already been applied and metadata exists.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information
            existing_meta_data: Existing metadata to merge with
            
        Returns:
            TimeSeriesDataset with merged metadata
        """
        logger.debug(f"Computing ARDS percentage for {dataset_type} dataset...")
        new_percentage_ards = dataset["ards"].sum() / len(dataset.index)
        logger.info(f"ARDS percentage in {dataset_type} set: {new_percentage_ards*100:.2f}%")
        
        logger.debug("Creating new metadata...")
        new_meta_data = TimeSeriesMetaDataManagement.factory_method(
            processing_meta_data, path, new_percentage_ards, dataset_type, additional_information
        )
        
        logger.debug("Merging existing and new metadata...")
        merged_meta_data = TimeSeriesMetaDataManagement.merge_meta_data(existing_meta_data, new_meta_data)
        logger.info(f"Successfully merged metadata for {dataset_type} dataset")
        
        return TimeSeriesDataset(content=dataset, meta_data=merged_meta_data)

    @staticmethod
    def _factory_new_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                              additional_information: str = None) -> TimeSeriesDataset:
        """
        Creates a TimeSeriesDataset with fresh metadata.
        Used when creating a dataset for the first time or without existing metadata.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information
            
        Returns:
            TimeSeriesDataset with new metadata
        """
        logger.debug(f"Computing ARDS percentage for {dataset_type} dataset...")
        percentage_ards = dataset["ards"].sum() / len(dataset.index)
        logger.info(f"ARDS percentage in {dataset_type} set: {percentage_ards*100:.2f}%")
        logger.info(f"Total samples in {dataset_type} set: {len(dataset.index)}")
        
        logger.debug("Creating new metadata object...")
        meta_data = TimeSeriesMetaDataManagement.factory_method(
            processing_meta_data, path, percentage_ards, dataset_type, additional_information
        )
        logger.info(f"Successfully created metadata for {dataset_type} dataset")
        
        return TimeSeriesDataset(content=dataset, meta_data=meta_data)

    @staticmethod
    def write(timeseries_dataset: TimeSeriesDataset):
        """
        Persists a TimeSeriesDataset to disk.
        Saves both the data content as CSV and the associated metadata as JSON.
        
        Args:
            timeseries_dataset: TimeSeriesDataset object to persist
        """
        dataset_type = timeseries_dataset.meta_data.dataset_type if hasattr(timeseries_dataset.meta_data, 'dataset_type') else "unknown"
        logger.info(f"Writing {dataset_type} dataset to disk...")
        
        path = timeseries_dataset.meta_data.dataset_location + ".csv"
        logger.info(f"Saving dataset content to: {path}")
        logger.debug(f"Dataset shape: {timeseries_dataset.content.shape}")
        
        try:
            timeseries_dataset.content.to_csv(path, index=False, header=True)
            logger.info(f"Dataset successfully saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to {path}: {str(e)}")
            raise
        
        logger.debug("Writing associated metadata...")
        try:
            TimeSeriesMetaDataManagement.write(timeseries_dataset.meta_data)
            logger.info(f"Metadata successfully written for {dataset_type} dataset")
        except Exception as e:
            logger.error(f"Failed to write metadata: {str(e)}")
            raise

class ImageDataset(Dataset):
    """Class to generate the dataset to run it in DL models"""
    
    def __init__(self, name, dl_method, disease, augment, path_datasets):
        """
        Generate dataset object.
        
        :param name: (str) Name of the dataset used
        :param dl_method: (str) Name of the DL method used
        :param disease: (str) Name of the diseases which are being used to classify with DL
        :param augment (bool) Whether we use the augmented version of the dataset or not
        """
        
        # initialise some varibles
        self.name = name
        self.dl_method = dl_method
        self.disease = disease
        self.augment = augment
        
        # get image and label path in a list
        self.path_dataset = path_datasets
        self.path = self.get_path(self.disease, self.name)
        self.folder_path = self.path
        
        # get all leaves of the path root
        self.path_walk_image, self.path_walk_label = self.child_node_path(self.path, self.name, self.augment)
        
        # initialise function for resizing and normalizing the images
        self.IMAGE_SIZE = self.get_image_size(self.dl_method)
        self.aug = albumentations.Compose([
           albumentations.Resize(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], interpolation=1, always_apply=True),
           albumentations.Normalize(mean=0.445, std=0.269, always_apply=True) # grayscale mean and std from ImageNet                               
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
        image = np.transpose(image,(2,0,1))
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
            path_image = os.path.join(self.path_dataset, name+'/image')
            path_label = os.path.join(self.path_dataset, name+'/label')
            path_image_augmented = os.path.join(self.path_dataset, name) 
            path_label_augmented = os.path.join(self.path_dataset, name) 
        elif disease == 'ARDS':
            path_image = os.path.join(self.path_dataset, name+'/image')
            path_label = os.path.join(self.path_dataset, name+'/label')
            path_image_augmented = os.path.join(self.path_dataset, name) 
            path_label_augmented = os.path.join(self.path_dataset, name) 
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
            for file in sorted(os.listdir(path[0]), key=lambda x:float(re.findall("(\d+)",x)[0])):
                path_walk_image.append(path[0]+"/"+file)
                path_walk_label.append(path[1]+"/"+file)
                
        # only include augmented image path if asked for it
        # for testset only include augmented image path without original/normal testset
        # for trainingset include augmented image path with original/normal trainingset
        if augment:
            for (dir_path, dir_names, file_names) in os.walk(path[2]):
                dir_names.sort()
                if len(file_names) != 0 and os.path.basename(dir_path) == 'image' and os.path.split(os.path.split(dir_path)[0])[1] in AUG_TECH:
                    for file in sorted(file_names, key=lambda x:float(re.findall("(\d+)",x)[0])):
                        path_walk_image.append(dir_path+"/"+file)
                if len(file_names) != 0 and os.path.basename(dir_path) == 'label' and os.path.split(os.path.split(dir_path)[0])[1] in AUG_TECH:
                    for file in sorted(file_names, key=lambda x:float(re.findall("(\d+)",x)[0])):
                        path_walk_label.append(dir_path+"/"+file)
                    
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

class DatasetGenerator:
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
        datasets_available_pneumonia = [name for name in os.listdir(path) if not name.startswith('.') and not name.endswith('.7z')]
        datasets_avaiable_ards = [name for name in os.listdir(path) if not name.startswith('.') and not name.endswith('.7z')]
        datasets_available = datasets_available_pneumonia + datasets_avaiable_ards
        
        # raise exception if dataset not available
        if not (dataset_name in datasets_available):
            raise Exception(str("Dataset is not supported. Supported datasets are: "+ ', '.join(datasets_available)))
            
        # generate Dataset Object with properties
        dataset = ImageDataset(dataset_name, dl_method, disease, augment, path)
        return dataset
