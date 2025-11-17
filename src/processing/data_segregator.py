import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math
import logging

logger = logging.getLogger(__name__)


class DataSegregator:
    """
    Handles data segregation into training and test sets with class balancing.
    Ensures proper distribution of ARDS and non-ARDS patients according to configured percentages.
    """

    def __init__(self, config):
        """
        Initializes the DataSegregator with configuration parameters.
        
        Args:
            config: Configuration dictionary containing training_test_ratio, percentage_of_ards_patients, and splitting_seed
            
        Raises:
            ValueError: If percentage_of_ards_patients is not specified
        """
        logger.info("Initializing DataSegregator...")
        self.training_test_ratio = config["training_test_ratio"]
        self.ards_percentage = config["percentage_of_ards_patients"]
        
        if not self.ards_percentage:
            error_msg = "Percentage of ARDS patients is required for data segregation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.random_state = config["splitting_seed"]
        logger.info(f"Training/Test ratio: {self.training_test_ratio}")
        logger.info(f"Target ARDS percentage: {self.ards_percentage * 100}%")
        logger.info(f"Random seed: {self.random_state}")
        logger.info("DataSegregator initialized successfully")
    
    def segregate_data(self, dataframe):
        """
        Segregates data into training and test sets with class balancing.
        Balances ARDS and non-ARDS samples to match target percentage, then splits into train/test.
        
        Args:
            dataframe: Input DataFrame containing patient data with 'ards', 'patient_id', and 'time' columns
            
        Returns:
            Tuple of (training_data, test_data) stratified by ARDS label
        """
        logger.info("=" * 80)
        logger.info("Starting Data Segregation and Class Balancing")
        logger.info("=" * 80)
        logger.info(f"Input data shape: {dataframe.shape}")
        
        random.seed(self.random_state)
        num_patients = dataframe.shape[0]
        
        # Separate ARDS and non-ARDS patients
        logger.info("Separating ARDS and non-ARDS patients...")
        ards_mask = dataframe["ards"] == 1
        ards_data = dataframe[ards_mask]
        non_ards_data = dataframe[~ards_mask]

        num_ards_patients = ards_data.shape[0]
        num_non_ards_patients = non_ards_data.shape[0]
        num_target_ards_patients = math.floor(self.ards_percentage * num_patients)
        num_target_non_ards_patients = math.floor((1 - self.ards_percentage) * num_patients)
        
        logger.info(f"Current distribution - ARDS: {num_ards_patients} ({num_ards_patients/num_patients*100:.1f}%), Non-ARDS: {num_non_ards_patients} ({num_non_ards_patients/num_patients*100:.1f}%)")
        logger.info(f"Target distribution - ARDS: {num_target_ards_patients} ({self.ards_percentage*100:.1f}%), Non-ARDS: {num_target_non_ards_patients} ({(1-self.ards_percentage)*100:.1f}%)")
        
        # Handle class imbalance
        if num_ards_patients > num_target_ards_patients:
            logger.warning("Too many ARDS patients - downsampling...")
            total_patients = math.floor(num_non_ards_patients * 1 / (1 - self.ards_percentage))
            num_target_ards_patients = math.floor(total_patients * self.ards_percentage)
            logger.info(f"Downsampling ARDS from {num_ards_patients} to {num_target_ards_patients} samples")
            
            samples = random.sample(range(num_ards_patients), num_target_ards_patients)
            ards_data_sampled = ards_data.iloc[samples]
            data = pd.concat([ards_data_sampled, non_ards_data]).reset_index(drop=True)
            logger.info(f"After balancing: ARDS={len(ards_data_sampled)}, Non-ARDS={len(non_ards_data)}")

        elif num_non_ards_patients > num_target_non_ards_patients:
            logger.warning("Too few ARDS patients - downsampling non-ARDS...")
            total_patients = math.floor(num_ards_patients * 1 / self.ards_percentage)
            num_target_non_ards_patients = math.floor(total_patients * (1 - self.ards_percentage))
            logger.info(f"Downsampling non-ARDS from {num_non_ards_patients} to {num_target_non_ards_patients} samples")
            
            samples = random.sample(range(num_non_ards_patients), num_target_non_ards_patients)
            non_ards_data_sampled = non_ards_data.iloc[samples]
            data = pd.concat([non_ards_data_sampled, ards_data]).reset_index(drop=True)
            logger.info(f"After balancing: ARDS={len(ards_data)}, Non-ARDS={len(non_ards_data_sampled)}")
        
        else:
            logger.info("Class distribution already balanced - no resampling needed")
            data = pd.concat([ards_data, non_ards_data], axis=0).reset_index(drop=True)
        
        logger.info(f"Balanced dataset shape: {data.shape}")
        
        # Remove metadata columns before train/test split
        logger.info("Removing metadata columns (time, patient_id)...")
        data.drop(["time", "patient_id"], axis=1, inplace=True)
        
        # Split into training and test sets with stratification
        logger.info(f"Splitting data into training ({(1-self.training_test_ratio)*100:.1f}%) and test ({self.training_test_ratio*100:.1f}%) sets...")
        training_data, test_data = train_test_split(
            data, 
            test_size=self.training_test_ratio, 
            random_state=self.random_state, 
            shuffle=True, 
            stratify=data["ards"]
        )
        
        train_ards_count = (training_data["ards"] == 1).sum()
        test_ards_count = (test_data["ards"] == 1).sum()
        
        logger.info("=" * 80)
        logger.info("Data Segregation Completed")
        logger.info(f"Training set: {training_data.shape[0]} samples (ARDS: {train_ards_count}, Non-ARDS: {training_data.shape[0]-train_ards_count})")
        logger.info(f"Test set: {test_data.shape[0]} samples (ARDS: {test_ards_count}, Non-ARDS: {test_data.shape[0]-test_ards_count})")
        logger.info("=" * 80)

        return training_data, test_data

    def set_training_test_ratio(self, new_ratio):
        """
        Updates the training/test split ratio at runtime.
        
        Args:
            new_ratio: New test set ratio (between 0 and 1)
        """
        logger.info(f"Updating training/test ratio from {self.training_test_ratio} to {new_ratio}")
        self.training_test_ratio = new_ratio

    def set_ards_percentage(self, new_percentage):
        """
        Updates the target ARDS percentage at runtime.
        
        Args:
            new_percentage: New target ARDS percentage (between 0 and 1)
        """
        logger.info(f"Updating ARDS percentage from {self.ards_percentage*100:.1f}% to {new_percentage*100:.1f}%")
        self.ards_percentage = new_percentage
