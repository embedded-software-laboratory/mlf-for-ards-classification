from processing.datasets_metadata import FilteringMetaData
import logging

logger = logging.getLogger(__name__)

class Filter:
    def __init__(self, config) -> None:
        """
        Initializes the Filter class with configuration settings.
        
        Args:
            config: Configuration dictionary containing filter settings
        """
        logger.info("Initializing Filter...")
        self.filter = []
        self.available_filter = ["Strict", "Lite", "BD"]
        self.set_filter(config["filter"])
        self.meta_data = None

    def create_meta_data(self):
        """
        Creates metadata for the applied filters if any filters are set.
        """
        if len(self.filter) > 0:
            logger.info("Creating metadata for applied filters...")
            self.meta_data = FilteringMetaData(applied_filters=self.filter)
            logger.info(f"Metadata created: {self.meta_data}")

    def filter_data(self, dataframe):
        """
        Applies the configured filters to the provided DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data to be filtered
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Starting data filtering process...")
        for filter_to_apply in self.filter:
            logger.debug(f"Applying filter: {filter_to_apply}")
            if filter_to_apply == "Strict":
                dataframe = self.filter_strict(dataframe)
            if filter_to_apply == "Lite":
                dataframe = self.filter_lite(dataframe)
            if filter_to_apply == "BD":
                dataframe = self.filter_BD(dataframe)
        return dataframe

    @staticmethod
    def filter_strict(dataframe):
        """
        Applies the strict filter to the DataFrame.
        Filters out patients without ARDS and with Horovitz < 200.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Filtered DataFrame
        """
        logger.debug("Applying strict filter...")
        patients_with_ards = dataframe.groupby("patient_id")["ards"].transform(lambda x: 1 in x.values)
        patients_without_ards = ~patients_with_ards
        patients_with_low_horovitz = dataframe.groupby("patient_id")["horovitz"].transform(lambda x: (x < 200).any())

        mask = ~(patients_without_ards & patients_with_low_horovitz)
        logger.debug(f"Strict filter applied. Remaining patients: {dataframe[mask].shape[0]}")
        return dataframe[mask]

    @staticmethod
    def filter_lite(dataframe):
        """
        Applies the lite filter to the DataFrame.
        Keeps patients with ARDS or those without all conditions (Horovitz < 200 and no comorbidities).
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Filtered DataFrame
        """
        logger.debug("Applying lite filter...")
        required_columns = ["hypervolemia", "pulmonary-edema", "heart-failure"]
        if not all(col in dataframe.columns for col in required_columns):
            logger.info("Skipping lite filter since not all necessary columns are present")
            return dataframe

        ards_mask = dataframe.groupby("patient_id")["ards"].transform(lambda x: 1 in x.values)
        horovitz_mask = dataframe.groupby("patient_id")["horovitz"].transform(lambda x: (x < 200).any())
        comorbidities_mask = ~dataframe[required_columns].eq(1).any(axis=1)

        # Keep rows where ARDS is present or where Horovitz < 200 and no comorbidity is present
        keep_mask = ards_mask | (horovitz_mask & comorbidities_mask)
        logger.debug(f"Lite filter applied. Remaining patients: {dataframe[keep_mask].shape[0]}")
        return dataframe[keep_mask]

    @staticmethod
    def filter_BD(dataframe):
        """
        Applies the BD filter to the DataFrame.
        Checks for patients with ARDS and ensures both PEEP >= 5 and Horovitz < 300 conditions are met.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Filtered DataFrame
        """
        logger.debug("Applying BD filter...")
        ards_mask = dataframe.groupby("patient_id")["ards"].transform(lambda x: 1 in x.values)

        # Condition that both PEEP >= 5 and Horovitz < 300 must be met
        peep_horovitz_mask = (dataframe["horovitz"] < 300) & (dataframe["peep"] >= 5)

        # Create a mask to keep patient IDs who satisfy both conditions in the same row
        valid_patient_ids = dataframe.groupby("patient_id").filter(
            lambda group: peep_horovitz_mask.loc[group.index].any()
        )["patient_id"].unique()

        # Keep rows where ARDS is not present or where PEEP and Horovitz conditions are satisfied
        keep_mask = ~ards_mask | dataframe["patient_id"].isin(valid_patient_ids)
        logger.debug(f"BD filter applied. Remaining patients: {dataframe[keep_mask].shape[0]}")
        return dataframe[keep_mask]

    def set_filter(self, filter_list):
        """
        Sets the filters to be applied based on the provided filter list.
        
        Args:
            filter_list: List of filters to apply
            
        Raises:
            RuntimeError: If any filter in the list is not available
        """
        logger.info("Setting filters...")
        for filter_to_apply in filter_list:
            if filter_to_apply not in self.available_filter:
                error_msg = f"Filter {filter_to_apply} not available. Available filters are: {self.available_filter}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        self.filter = filter_list
        logger.info(f"Filters successfully set: {self.filter}")
