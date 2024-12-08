from processing.datasets_metadata import FilteringMetaData

class Filter:
    def __init__(self, config) -> None:
        self.filter = []
        self.available_filter = ["Strict", "Lite", "BD"]
        self.set_filter(config["filter"])
        self.meta_data = None

    def create_meta_data(self):
        if len(self.filter) > 0:
            self.meta_data = FilteringMetaData(applied_filters=self.filter)

    def filter_data(self, dataframe):

        for filter_to_apply in self.filter:
            if filter_to_apply == "Strict":
                dataframe = self.filter_strict(dataframe)
            if filter_to_apply == "Light":
                dataframe = self.filter_lite(dataframe)
            if filter_to_apply == "BD":
                dataframe = self.filter_BD(dataframe)
        return dataframe

    @staticmethod
    def filter_strict(dataframe):
        # Filter patients without ARDS and Horovitz < 200
        patients_with_ards = dataframe.groupby(
            "patient_id")["ards"].transform(lambda x: 1 in x.values)
        patients_without_ards = ~patients_with_ards
        patients_with_low_horovitz = dataframe.groupby(
            "patient_id")["horovitz"].transform(lambda x: (x < 200).any())

        mask = ~(patients_without_ards & patients_with_low_horovitz)
        return dataframe[mask]

    @staticmethod
    def filter_lite(dataframe):
        # Keep patients with ARDS or without all of the conditions (Horovitz < 200 and no comorbidities)
        required_columns = ["hypervolemia", "pulmonary-edema", "heart-failure"]
        if not all(col in dataframe.columns for col in required_columns):
            print("Skipping filter b since not all necessary columns are present")
            return dataframe

        ards_mask = dataframe.groupby("patient_id")[
            "ards"].transform(lambda x: 1 in x.values)
        horovitz_mask = dataframe.groupby(
            "patient_id")["horovitz"].transform(lambda x: (x < 200).any())
        comorbidities_mask = ~dataframe[required_columns].eq(1).any(axis=1)

        # Keep rows where ARDS is present or where Horovitz < 200 and no comorbidity is present
        keep_mask = ards_mask | (horovitz_mask & comorbidities_mask)
        return dataframe[keep_mask]

    @staticmethod
    def filter_BD(dataframe):
        # Check for patients with ARDS
        ards_mask = dataframe.groupby("patient_id")[
            "ards"].transform(lambda x: 1 in x.values)

        # Condition that both PEEP >= 5 and Horovitz < 300 must be met
        peep_horovitz_mask = (dataframe["horovitz"] < 300) & (
            dataframe["peep"] >= 5)

        # Create a mask to keep patient IDs who satisfy both conditions in the same row
        valid_patient_ids = dataframe.groupby("patient_id").filter(
            lambda group: peep_horovitz_mask.loc[group.index].any()
        )["patient_id"].unique()

        # Keep rows where ARDS is not present or where PEEP and Horovitz conditions are satisfied
        keep_mask = ~ards_mask | dataframe["patient_id"].isin(
            valid_patient_ids)
        return dataframe[keep_mask]

    def set_filter(self, filter_list):
        for filter_to_apply in filter_list:
            if filter_to_apply not in self.available_filter:
                raise RuntimeError(
                    "Filter " + filter_to_apply + " not available. Avaliable are " + str(self.available_filter))
        self.filter = filter_list
