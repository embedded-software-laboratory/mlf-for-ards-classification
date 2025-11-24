import pandas as pd
import logging

from processing.datasets_metadata import UnitConversionMetaData

logger = logging.getLogger(__name__)

class UnitConverter:
    """
    Handles unit conversion for clinical parameters across different databases.
    Converts parameters from source units to standard units based on database-specific formulas.
    """

    def __init__(self):
        """
        Initializes the UnitConverter with database-specific conversion formulas.
        """
        logger.info("Initializing UnitConverter...")
        self._conversion_formulas = {
            "eICU": {"creatinine": self.convert_creatinine, "crp": self.convert_crp,
                     "bilirubin": self.convert_bilirubin, "etco2": self.convert_etco2, "bnp": self.convert_bnp},
            "MIMIC3": {"Harnstoff": self.convert_harnstoff,
                       "creatinine": self.convert_creatinine, "crp": self.convert_crp,
                       "bilirubin": self.convert_bilirubin, "bnp": self.convert_bnp},
            "MIMIC4": {"creatinine": self.convert_creatinine, "crp": self.convert_crp, "bnp": self.convert_bnp},
            "UKA": {"hemoglobin": self.convert_hemoglobin, "albumin": self.convert_albumin, "bnp": self.convert_bnp, "crp": self.convert_crp},
            "CALIBRATION": {"fio2": self.convert_fio2, "hemoglobin": self.convert_hemoglobin, "albumin": self.convert_albumin, "bnp": self.convert_bnp, "crp": self.convert_crp},
            "CONTROL": {"hemoglobin": self.convert_hemoglobin, "albumin": self.convert_albumin, "bnp": self.convert_bnp, "crp": self.convert_crp}}
        self._columns_to_convert = None
        self.meta_data = None
        logger.info("UnitConverter initialized successfully.")

    def convert_units(self, dataframe: pd.DataFrame, database_name: str, job_number: int, total_job_count: int):
        """
        Converts units for specified columns in the DataFrame using database-specific formulas.
        
        Args:
            dataframe: DataFrame containing patient data with columns to convert
            database_name: Name of the database (e.g., 'MIMIC3', 'eICU', 'UKA')
            job_number: Current job number for logging
            total_job_count: Total number of jobs for logging
            
        Returns:
            DataFrame with converted units
        """
        logger.info(f"Start unit conversion for job {job_number} of {total_job_count} jobs...")
        logger.debug(f"Database: {database_name}, Columns to convert: {self._columns_to_convert}")
        
        if database_name not in self.conversion_formulas:
            logger.error(f"Database '{database_name}' not supported for unit conversion")
            raise ValueError(f"Database '{database_name}' not supported")
        
        if not self._columns_to_convert:
            logger.warning("No columns to convert specified")
            logger.info(f"Finished unit conversion for job {job_number} of {total_job_count} jobs...")
            return dataframe
        
        columns_converted = 0
        for column in self._columns_to_convert:
            if column in dataframe.columns:
                logger.debug(f"Converting column: {column}")
                dataframe[column] = dataframe[column].apply(self.conversion_formulas[database_name][column])
                columns_converted += 1
            else:
                logger.warning(f"Column '{column}' not found in dataframe. Skipping conversion.")
        
        logger.info(f"Converted {columns_converted} columns")
        logger.info(f"Finished unit conversion for job {job_number} of {total_job_count} jobs...")
        return dataframe

    def create_meta_data(self, database_name: str):
        """
        Creates metadata describing the unit conversions applied.
        
        Args:
            database_name: Name of the database to create metadata for
        """
        logger.info("Creating unit conversion metadata...")
        
        if not self._columns_to_convert:
            logger.warning("No columns to convert - metadata will be empty")
            self.meta_data = UnitConversionMetaData(conversions={})
            return
        
        meta_data_dict = {}
        for column in self._columns_to_convert:
            if column in self.conversion_formulas.get(database_name, {}):
                conversion_func = self.conversion_formulas[database_name][column]
                # Store function name as the conversion description
                meta_data_dict[column] = conversion_func.__name__
                logger.debug(f"Metadata added for {column}: {conversion_func.__name__}")
            else:
                logger.warning(f"Conversion formula not found for {column} in {database_name}")
        
        self.meta_data = UnitConversionMetaData(conversions=meta_data_dict)
        logger.info(f"Metadata created with {len(meta_data_dict)} conversion records")

    @property
    def conversion_formulas(self):
        """
        Returns the dictionary of conversion formulas organized by database.
        
        Returns:
            Dictionary mapping database names to conversion formula dictionaries
        """
        return self._conversion_formulas

    @staticmethod
    def convert_bnp(value):
        """
        Converts B-type natriuretic peptide (BNP) using factor 0.1182 (from pg/dL to pmol/L).
        
        Args:
            value: BNP value to convert
            
        Returns:
            Converted BNP value
        """
        return value * 0.1182

    @staticmethod
    def convert_hemoglobin(value):
        """
        Converts hemoglobin from g/dL to g/L using factor 0.6206.
        
        Args:
            value: Hemoglobin value to convert
            
        Returns:
            Converted hemoglobin value
        """
        return value * 0.0621

    @staticmethod
    def convert_creatinine(value):
        """
        Converts creatinine from mg/dL to µmol/L using factor * 88.4017
        
        Args:
            value: Creatinine value to convert
            
        Returns:
            Converted creatinine value
        """
        return value * 88.4017

    @staticmethod
    def convert_harnstoff(value):
        """
        Converts Harnstoff (urea) using factor 0.1665.
        
        Args:
            value: Harnstoff value to convert
            
        Returns:
            Converted Harnstoff value
        """
        return value * 0.1665

    @staticmethod
    def convert_albumin(value):
        """
        Converts albumin using factor 151.5152.
        
        Args:
            value: Albumin value to convert
            
        Returns:
            Converted albumin value
        """
        return value * 0.06646

    @staticmethod
    def convert_crp(value):
        """
        Converts C-reactive protein (CRP) using factor 9.5238.
        
        Args:
            value: CRP value to convert
            
        Returns:
            Converted CRP value
        """
        return value * 9.5238

    @staticmethod
    def convert_bilirubin(value):
        """
        Converts bilirubin using factor 17.1037.
        
        Args:
            value: Bilirubin value to convert
            
        Returns:
            Converted bilirubin value
        """
        return value * 17.1037

    @staticmethod
    def convert_etco2(value):
        """
        Converts end-tidal CO2 (etCO2) using factor 1.7.
        
        Args:
            value: EtCO2 value to convert
            
        Returns:
            Converted EtCO2 value
        """
        return value * 1.7
    
    @staticmethod
    def convert_fio2(value):
        """
        Converts fraction of inspired oxygen (FiO2) checking for lower values.
        
        Args:
            value: FiO2 value to convert
            
        Returns:
            Converted FiO2 value
        """
        if value <= 1.0:
            value = value * 100.0
        return value

    @property
    def columns_to_convert(self):
        """
        Returns the list of columns that will be converted.
        
        Returns:
            List of column names to convert
        """
        return self._columns_to_convert

    @columns_to_convert.setter
    def columns_to_convert(self, columns_to_convert):
        """
        Sets the columns to be converted during unit conversion.
        
        Args:
            columns_to_convert: List of column names to convert
        """
        logger.info(f"Setting columns to convert: {columns_to_convert}")
        self._columns_to_convert = columns_to_convert
        logger.debug(f"Columns to convert updated. Count: {len(columns_to_convert) if columns_to_convert else 0}")

