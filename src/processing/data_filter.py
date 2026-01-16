from processing.datasets_metadata import FilteringMetaData
import logging

logger = logging.getLogger(__name__)

class DataFilter:
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
        initial_patients = dataframe['patient_id'].nunique()
        initial_ards_patients = dataframe[dataframe['ards'] == 1]['patient_id'].nunique()
        logger.info(f"Initial: {initial_patients} patients total, {initial_ards_patients} ARDS patients")
        
        for filter_to_apply in self.filter:
            pre_filter_patients = dataframe['patient_id'].nunique()
            pre_filter_ards = dataframe[dataframe['ards'] == 1]['patient_id'].nunique()
            
            if filter_to_apply == "Strict":
                dataframe = self.filter_strict(dataframe)
            elif filter_to_apply == "Lite":
                dataframe = self.filter_lite(dataframe)
            elif filter_to_apply == "BD":
                dataframe = self.filter_BD(dataframe)
            
            post_filter_patients = dataframe['patient_id'].nunique()
            post_filter_ards = dataframe[dataframe['ards'] == 1]['patient_id'].nunique()
            excluded_total = pre_filter_patients - post_filter_patients
            excluded_ards = pre_filter_ards - post_filter_ards
            
            logger.info(f"After {filter_to_apply} filter: {post_filter_patients} patients total ({excluded_total} excluded), {post_filter_ards} ARDS patients ({excluded_ards} ARDS excluded)")
        
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

        # treat placeholder values as missing
        filtered_horo = dataframe["horovitz"].where(dataframe["horovitz"] > -100000.0)

        patients_with_ards = dataframe.groupby("patient_id")["ards"].transform(lambda x: 1 in x.values)
        patients_without_ards = ~patients_with_ards
        patients_with_low_horo = filtered_horo.groupby(dataframe["patient_id"]).transform(lambda x: (x < 200).any())

        # Patients to exclude: no ARDS AND low Horovitz
        exclude_mask = patients_without_ards & patients_with_low_horo
        excluded_patients = dataframe[exclude_mask]['patient_id'].nunique()
        logger.debug(f"Strict filter: Excluding {excluded_patients} patients without ARDS who have Horovitz < 200")

        mask = ~(patients_without_ards & patients_with_low_horo)

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

        # treat placeholder values as missing
        filtered_horo = dataframe["horovitz"].where(dataframe["horovitz"] > -100000.0)

        ards_mask = dataframe.groupby("patient_id")["ards"].transform(lambda x: 1 in x.values)
        horovitz_mask = filtered_horo.groupby(dataframe["patient_id"]).transform(lambda x: (x < 200).any())
        comorbidities_mask = ~dataframe[required_columns].eq(1).any(axis=1)

        # Keep: ARDS patients OR (low Horovitz AND no comorbidities)
        keep_mask = ards_mask | (horovitz_mask & comorbidities_mask)
        
        excluded_patients = dataframe[~keep_mask]['patient_id'].nunique()
        excluded_ards = dataframe[~keep_mask & ards_mask]['patient_id'].nunique()
        logger.debug(f"Lite filter: Excluding {excluded_patients} patients total, {excluded_ards} ARDS patients")
        if excluded_ards > 0:
            logger.warning(f"Lite filter: Excluding {excluded_ards} ARDS patients who have low Horovitz and comorbidities")

        valid_patient_ids = (
            dataframe.groupby("patient_id")
            .filter(lambda group: peep_horo_mask.loc[group.index].any())
            ["patient_id"]
            .unique()
        )
        
        valid_ards_patients = len([pid for pid in valid_patient_ids if dataframe[dataframe['patient_id'] == pid]['ards'].any()])
        logger.debug(f"BD filter: {len(valid_patient_ids)} patients meet PEEP>=5 and Horovitz<300 criteria, {valid_ards_patients} are ARDS patients")

        keep_mask = ~ards_mask | dataframe["patient_id"].isin(valid_patient_ids)
        
        excluded_ards = ards_patients - valid_ards_patients
        if excluded_ards > 0:
            logger.warning(f"BD filter: Excluding {excluded_ards} ARDS patients who don't meet PEEP>=5 and Horovitz<300 criteria")
        
        return dataframe[keep_mask]

    def set_filter(self, filter_list):
        """
        Sets the filters to be applied based on the provided filter list.
        
        Args:
            filter_list: List of filters to apply
            
        Raises:
            RuntimeError: If any filter in the list is not available
        """
        for filter_to_apply in filter_list:
            if filter_to_apply not in self.available_filter:
                error_msg = f"Filter {filter_to_apply} not available. Available filters are: {self.available_filter}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        self.filter = filter_list
        logger.info(f"Filters successfully set: {self.filter}")
