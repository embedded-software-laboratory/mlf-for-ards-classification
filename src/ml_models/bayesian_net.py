from ml_models.model_interface import Model
from ml_models.timeseries_model import TimeSeriesModel
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork
import numpy as np
import pomegranate
import json

class Bayesian_network(TimeSeriesModel):

    def __init__(self):
        super().__init__()
        self.name = "Bayesian Network"
        self.model = self.make_bn()

    def train_model(self, training_data):
        """Function that starts the learning process of the SVM and stores the resulting model after completion"""
        
        training_data = training_data.drop(columns=['patient_id'])
        converted_data = self.convert_dataset(training_data)
        # Init forest and read training data
        self.model.fit(converted_data, pseudocount=1)
        self.model.bake()

    def predict(self, dataframe):
        converted_datasets = self.convert_dataset(dataframe)
        prediction = self.model.predict(converted_datasets)
        result = []
        for p in prediction:
            result.append(p[0])
        return result
    
    def predict_proba(self, data):
        converted_datasets = self.convert_dataset(data)
        prediction = self.model.predict_proba(converted_datasets)
        result = []
        for p in prediction:
            temp = []
            temp.append(p[0].probability(0))
            temp.append(p[0].probability(1))
            result.append(temp)
        return np.array(result)

    def has_predict_proba(self):
        return True
    
    def save(self, filepath):
        file = open(filepath + ".json", "w")
        json.dump(self.model.to_json(), file)
    
    def load(self, filepath):
        file = open(filepath + ".json", "r")
        self.model = pomegranate.from_json(json.load(file))
    
    def convert_dataset(self, data):
        converted_datasets = []
        for i in data.index:
            converted_datasets.append(self.convert_single_dataset(data.loc[i]))
        return converted_datasets
    
    def convert_single_dataset(self, data): 
        array = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
        for key in data.keys():
            if key == "compliance":
                if data["compliance"] > 40:
                    array[1] = 0
                else:
                    array[1] = 1
            if key == "af":
                if data["af"] > 20:
                    array[2] = 0
                else:
                    array[2] = 1
            if key == "be": 
                if data["be"] > 2:
                    array[3] = 0
                elif data["be"] <= 2 and data["be"] >= -2:
                    array[3] = 1
                else:
                    array[3] = 2
            if key == "albumin":
                if data["albumin"] >= 3.5:
                    array[4] = 0
                else:
                    array[4] = 1
            if key == "bilirubin":
                if data["bilirubin"] >= 20:
                    array[5] = 0
                else:
                    array[5] = 1
            if key == "bnp":
                if data["bnp"] > 400:
                    array[6] = 0
                else:
                    array[6] = 1
            if key == "ck":
                if data["ck"] > 190:
                    array[7] = 0
                else:
                    array[7] = 1
            if key == "ck-mb":
                if data["ck-mb"] >= 6:
                    array[8] = 0
                else:
                    array[8] = 1
            if key == "hemoglobin":
                if data["hemoglobin"] > 10:
                    array[9] = 0
                else:
                    array[9] = 1
            if key == "hf":
                if data["hf"] >= 90:
                    array[10] = 0
                else:
                    array[10] = 1
            if key == "core-temp":
                if data["core-temp"] > 38:
                    array[11] = 0
                elif data["core-temp"] <= 38 and data["core-temp"] >=36 :
                    array[11] = 1
                else:
                    array[11] = 2
            if key == "creatinine":
                if data["creatinine"] >= 110:
                    array[12] = 0
                else:
                    array[12] = 1
            if key == "p-ei":
                if data["p-ei"] > 30:
                    array[13] = 0
                elif data["p-ei"] <= 30:
                    array[13] = 1
                else:
                    array[13] = 2
            if key == "paco2":
                if data["paco2"] >= 46:
                    array[14] = 0
                else:
                    array[14] = 1
            if key == "pao2":
                if data["pao2"] >= 75:
                    array[15] = 0
                else:
                    array[15] = 1
            if key == "peep":
                if data["peep"] > 15:
                    array[16] = 0
                elif data["peep"] <= 15 and data["peep"] >= 10:
                    array[16] = 1
                elif data["peep"] < 10 and data["peep"] >= 5:
                    array[16] = 2
                elif data["peep"] < 5:
                    array[16] = 3
                else:
                    array[16] = 4
            if key == "ph-art":
                if data["ph-art"] >= 7.45:
                    array[17] = 0
                elif data["ph-art"] < 7.45 and data["ph-art"] >=7.37:
                    array[17] = 1
                else:
                    array[17] = 2
            if key == "spo2":
                if data["spo2"] >= 94:
                    array[18] = 0
                else:
                    array[18] = 1
            if key == "thrombocytes":
                if data["thrombocytes"] >= 140:
                    array[19] = 0
                else:
                    array[19] = 1
            if key == "troponin":
                if data["troponin"] >= 0.05:
                    array[20] = 0
                else:
                    array[20] = 1
            if key == "bicarbonate":
                if data["bicarbonate"] > 26:
                    array[21] = 0
                elif data["bicarbonate"] <= 26 and data["bicarbonate"] >= 22:
                    array[21] = 0
                else:
                    array[21] = 1
            if key == "lactate":
                if data["lactate"] > 2:
                    array[22] = 0
                else:
                    array[22] = 1
            if key == "leukocytes":
                if data["leukocytes"] > 12:
                    array[23] = 0
                elif data["leukocytes"] <= 12 and  data["leukocytes"] >= 4:
                    array[23] = 1
                else:
                    array[23] = 2
            if key == "liquid-balance":
                if data["liquid-balance"] > 1000:
                    array[24] = 0
                elif data["liquid-balance"] <= 1000 and data["liquid-balance"] >= -1000:
                    array[24] = 1
                else:
                    array[24] = 2
            if key == "delta-p":
                if data["delta-p"] >= 15:
                    array[25] = 0
                elif data["delta-p"] < 15:
                    array[25] = 1
                else:
                    array[25] = 2
            if key == "got":
                if data["got"] >= 38:
                    array[26] = 0
                else:
                    array[26] = 1
            if key == "gpt":
                if data["gpt"] >= 41:
                    array[27] = 0
                else:
                    array[27] = 1
            if key == "urea":
                if data["urea"] > 8.3:
                    array[28] = 0
                else:
                    array[28] = 1
            if key == "heart-failure":
                if data["heart-failure"] == 0:
                    array[29] = 0
                else:
                    array[29] = 1
            if key == "horovitz":
                if data["horovitz"] < 100:
                    array[30] = 0
                elif data["horovitz"] >= 100 and data["horovitz"] < 200:
                    array[30] = 1
                elif data["horovitz"] >= 200 and data["horovitz"] < 300:
                    array[30] = 2
                else:
                    array[30] = 3
            if key == "hypervolemia":
                if data["hypervolemia"] == 0:
                    array[31] = 0
                else:
                    array[31] = 1
            if key == "tidal-vol-per-kg":
                if data["tidal-vol-per-kg"] > 6:
                    array[32] = 0
                elif data["tidal-vol-per-kg"] <= 6:
                    array[32] = 1
                else:
                    array[32] = 2
            if key == "pulmonary-edema":
                if data["pulmonary-edema"] == 0:
                    array[33] = 0
                else:
                    array[33] = 1
            if key == "lymphocytes":
                if data["lymphocytes"] > 50:
                    array[34] = 0
                elif data["lymphocytes"] <= 50 and data["lymphocytes"] >= 15:
                    array[34] = 0
                else:
                    array[34] = 1
            if key == "mech-vent":
                if data["mech-vent"] == 0:
                    array[35] = 0
                else:
                    array[35] = 1
            if key == "nt-per-bnp":
                if data["nt-per-bnp"] >= 450:
                    array[36] = 0
                else:
                    array[36] = 1
            if key == "pneumonia":
                if data["pneumonia"] == 0:
                    array[37] = 0
                else:
                    array[37] = 1
            if key == "xray":
                if data["xray"] == 0:
                    array[38] = 0
                else:
                    array[38] = 1
            if key == "sepsis":
                if data["sepsis"] == 0:
                    array[39] = 0
                else:
                    array[39] = 1
            if key == "chest-injury":
                if data["chest-injury"] == 0:
                    array[40] = 0
                else:
                    array[40] = 1
    
        return array
    

    def make_bn(self):
        Thoraxtrauma = DiscreteDistribution({1: 0.02, 0: 0.98})
        Pneumonie = DiscreteDistribution({1: 0.16, 0: 0.84})
        Sepsis = DiscreteDistribution({1: 0.126, 0: 0.874})
        Thrombozyten = ConditionalProbabilityTable(
            [[1, 1, 0.28], [1, 0, 0.72],
            [0, 1, 0.2], [0, 0, 0.8]],
            [Sepsis])

        Bilirubin = ConditionalProbabilityTable(
            [[1, 1, 0.69], [1, 0, 0.31],
            [0, 1, 0.81], [0, 0, 0.19]],
            [Sepsis])

        GOT = ConditionalProbabilityTable(
            [[1, 1, 0.57], [1, 0, 0.43],
            [0, 1, 0.62], [0, 0, 0.38]],
            [Sepsis])

        GPT = ConditionalProbabilityTable(
            [[1, 1, 0.68], [1, 0, 0.32],
            [0, 1, 0.73], [0, 0, 0.27]],
            [Sepsis])

        Leukozyten = ConditionalProbabilityTable(
            [[1, 2, 0.07], [1, 1, 0.49], [1, 0, 0.44],
            [0, 2, 0.04], [0, 1, 0.66], [0, 0, 0.30]],
            [Sepsis])

        Lymphozyten = ConditionalProbabilityTable(
            [[2, 2, 0.36], [2, 1, 0.56], [2, 0, 0.08],
            [1, 2, 0.49], [1, 1, 0.5], [1, 0, 0.01],
            [0, 2, 0.76], [0, 1, 0.23], [0, 0, 0.01]],
            [Leukozyten])

        Albunin = ConditionalProbabilityTable(
            [[1, 1, 0.895], [1, 0, 0.105],
            [0, 1, 0.643], [0, 0, 0.357]],
            [Sepsis])

        Korperkerntemperatur = ConditionalProbabilityTable(
            [[1, 2, 0.14], [1, 1, 0.79], [1, 0, 0.07],
            [0, 2, 0.09], [0, 1, 0.87], [0, 0, 0.04]],
            [Sepsis])

        ards = ConditionalProbabilityTable(
            [[1, 1, 1, 1, 0.24], [1, 1, 1, 0, 0.76],
            [0, 1, 1, 1, 0.23], [0, 1, 1, 0, 0.77],
            [1, 0, 1, 1, 0.23], [1, 0, 1, 0, 0.77],
            [0, 0, 1, 1, 0.21], [0, 0, 1, 0, 0.79],
            [1, 1, 0, 1, 0.27], [1, 1, 0, 0, 0.73],
            [0, 1, 0, 1, 0.26], [0, 1, 0, 0, 0.74],
            [1, 0, 0, 1, 0.21], [1, 0, 0, 0, 0.79],
            [0, 0, 0, 1, 0.22], [0, 0, 0, 0, 0.78]],
            [Sepsis, Thoraxtrauma, Pneumonie])

        h24Bilanz = DiscreteDistribution({2: 0.22, 1: 0.52, 0: 0.26})

        Hypervolamie = ConditionalProbabilityTable(
            [[1, 2, 1, 0.14], [1, 2, 0, 0.86],
            [1, 1, 1, 0.13], [1, 1, 0, 0.87],
            [1, 0, 1, 0.33], [1, 0, 0, 0.67],
            [0, 2, 1, 0.03], [0, 2, 0, 0.97],
            [0, 1, 1, 0.02], [0, 1, 0, 0.98],
            [0, 0, 1, 0.24], [0, 0, 0, 0.76]],
            [Sepsis, h24Bilanz])

        Herzversagen = DiscreteDistribution({1: 0.27, 0: 0.73})

        BNP = ConditionalProbabilityTable(
            [[1, 1, 0.02], [1, 0, 0.98],
            [0, 1, 0.05], [0, 0, 0.95]],
            [Herzversagen])

        NTproBNP = ConditionalProbabilityTable(
            [[1, 1, 0.15], [1, 0, 0.85],
            [0, 1, 0.01], [0, 0, 0.99]],
            [BNP])

        Troponin = ConditionalProbabilityTable(
            [[1, 1, 0.37], [1, 0, 0.63],
            [0, 1, 0.58], [0, 0, 0.42]],
            [Herzversagen])

        CK = ConditionalProbabilityTable(
            [[1, 1, 0.77], [1, 0, 0.23],
            [0, 1, 0.67], [0, 0, 0.33]],
            [Herzversagen])

        CKMB = ConditionalProbabilityTable(
            [[1, 1, 0.41], [1, 0, 0.59],
            [0, 1, 0.77], [0, 0, 0.23]],
            [CK])

        Lungenodem = ConditionalProbabilityTable(
            [[1, 1, 1, 0.36], [1, 1, 0, 0.64],
            [0, 1, 1, 0.23], [0, 1, 0, 0.77],
            [1, 0, 1, 0.11], [1, 0, 0, 0.89],
            [0, 0, 1, 0.01], [0, 0, 0, 0.99]],
            [Herzversagen, Hypervolamie])

        HF = ConditionalProbabilityTable(
            [[1, 1, 1, 0.62], [1, 1, 0, 0.38],
            [0, 1, 1, 0.57], [0, 1, 0, 0.43],
            [1, 0, 1, 0.68], [1, 0, 0, 0.32],
            [0, 0, 1, 0.65], [0, 0, 0, 0.35]],
            [Herzversagen, ards])

        SpO2 = ConditionalProbabilityTable(
            [[1, 1, 0.02], [1, 0, 0.98],
            [0, 1, 0.02], [0, 0, 0.98]],
            [ards])

        paO2 = ConditionalProbabilityTable(
            [[1, 1, 0.04], [1, 0, 0.96],
            [0, 1, 0.04], [0, 0, 0.96]],
            [ards])

        paCO2 = ConditionalProbabilityTable(
            [[1, 1, 0.73], [1, 0, 0.27],
            [0, 1, 0.75], [0, 0, 0.25]],
            [ards])

        Hamoglobin = ConditionalProbabilityTable(
            [[1, 1, 0.2], [1, 0, 0.8],
            [0, 1, 0.42], [0, 0, 0.58]],
            [ards])

        Rontgenbild = ConditionalProbabilityTable(
            [[1, 1, 1, 0.99], [1, 1, 0, 0.01],
            [0, 1, 1, 0.99], [0, 1, 0, 0.01],
            [1, 0, 1, 0.99], [1, 0, 0, 0.01],
            [0, 0, 1, 0.01], [0, 0, 0, 0.99]],
            [Lungenodem, ards])

        Kreatinin = ConditionalProbabilityTable(
            [[1, 1, 1, 0.59], [1, 1, 0, 0.41],
            [0, 1, 1, 0.68], [0, 1, 0, 0.32],
            [1, 0, 1, 0.57], [1, 0, 0, 0.43],
            [0, 0, 1, 0.76], [0, 0, 0, 0.24]],
            [Sepsis, ards])

        Harnstoff = ConditionalProbabilityTable(
            [[1, 1, 1, 0.6], [1, 1, 0, 0.4],
            [0, 1, 1, 0.002], [0, 1, 0, 0.998],
            [1, 0, 1, 0.002], [1, 0, 0, 0.998],
            [0, 0, 1, 0.002], [0, 0, 0, 0.998]],
            [Sepsis, ards])

        # Beatmung
        MaschinelleBeatmung = ConditionalProbabilityTable(
            [[1, 1, 0.7], [1, 0, 0.3],
            [0, 1, 0.48], [0, 0, 0.52]],
            [ards])

        individuellesVT = ConditionalProbabilityTable(
            [[1, 2, 0.05], [1, 1, 0.18], [1, 0, 0.82],
            [0, 2, 0.99], [0, 1, 0.005], [0, 0, 0.005]],
            [MaschinelleBeatmung])

        PEI = ConditionalProbabilityTable(
            [[1, 2, 0.0003], [1, 1, 0.8808], [1, 0, 0.1189],
            [0, 2, 0.9996], [0, 1, 0.0002], [0, 0, 0.0002]],
            [MaschinelleBeatmung])

        Atemfrequenz = ConditionalProbabilityTable(
            [[1, 1, 0.89], [1, 0, 0.11],
            [0, 1, 0.5], [0, 0, 0.5]],
            [MaschinelleBeatmung])

        Compliance = ConditionalProbabilityTable(
            [[1, 1, 1, 0.89], [1, 1, 0, 0.11],
            [0, 1, 1, 0.5], [0, 1, 0, 0.5],
            [1, 0, 1, 0.82], [1, 0, 0, 0.18],
            [0, 0, 1, 0.74], [0, 0, 0, 0.26]],
            [MaschinelleBeatmung, ards])

        PEEP = ConditionalProbabilityTable(
            [[1, 4, 0.0002], [1, 3, 0.085], [1, 2, 0.833], [1, 1, 0.073], [1, 0, 0.009],
            [0, 4, 0.9992], [0, 3, 0.0002], [0, 2, 0.0002], [0, 1, 0.0002], [0, 0, 0.0002]],
            [MaschinelleBeatmung])

        Horovitzquotient = ConditionalProbabilityTable(
            [[1, 1, 3, 0.02], [1, 1, 2, 0.02], [1, 1, 1, 0.37], [1, 1, 0, 0.59],
            [0, 1, 3, 0.03], [0, 1, 2, 0.05], [0, 1, 1, 0.4], [0, 1, 0, 0.52],
            [1, 0, 3, 0.34], [1, 0, 2, 0.33], [1, 0, 1, 0.28], [1, 0, 0, 0.05],
            [0, 0, 3, 0.19], [0, 0, 2, 0.29], [0, 0, 1, 0.4], [0, 0, 0, 0.12]],
            [MaschinelleBeatmung, ards])

        deltaP = ConditionalProbabilityTable(
            [[4, 2, 2, 0.99], [4, 2, 1, 0.005], [4, 2, 0, 0.005],
            [4, 1, 2, 0.99], [4, 1, 1, 0.005], [4, 1, 0, 0.005],
            [4, 0, 2, 0.99], [4, 0, 1, 0.005], [4, 0, 0, 0.005],
            [3, 2, 2, 0.99], [3, 2, 1, 0.005], [3, 2, 0, 0.005],
            [3, 1, 2, 0.01], [3, 1, 1, 0.52], [3, 1, 0, 0.47],
            [3, 0, 2, 0.01], [3, 0, 1, 0.01], [3, 0, 0, 0.98],
            [2, 2, 2, 0.99], [2, 2, 1, 0.005], [2, 2, 0, 0.005],
            [2, 1, 2, 0.01], [2, 1, 1, 0.52], [2, 1, 0, 0.47],
            [2, 0, 2, 0.01], [2, 0, 1, 0.01], [2, 0, 0, 0.98],
            [1, 2, 2, 0.99], [1, 2, 1, 0.005], [1, 2, 0, 0.005],
            [1, 1, 2, 0.1], [1, 1, 1, 0.1], [1, 1, 0, 0.8],
            [1, 0, 2, 0.01], [1, 0, 1, 0.01], [1, 0, 0, 0.98],
            [0, 2, 2, 0.99], [0, 2, 1, 0.005], [0, 2, 0, 0.005],
            [0, 1, 2, 0.01], [0, 1, 1, 0.95], [0, 1, 0, 0.03],
            [0, 0, 2, 0.01], [0, 0, 1, 0.2], [0, 0, 0, 0.79]],
            [PEEP, PEI])

        # PH
        Laktat = DiscreteDistribution({1: 0.26, 0: 0.74})
        Bicarbonat = DiscreteDistribution({2: 0.44, 1: 0.42, 0: 0.14})
        BE = DiscreteDistribution({2: 0.24, 1: 0.47, 0: 0.29})
        ph = ConditionalProbabilityTable(
            [[1, 2, 2, 1, 1, 2, 0.4], [1, 2, 2, 1, 1, 1, 0.4], [1, 2, 2, 1, 1, 0, 0.2],
            [1, 1, 2, 1, 1, 2, 0.4], [1, 1, 2, 1, 1, 1, 0.4], [1, 1, 2, 1, 1, 0, 0.2],
            [1, 0, 2, 1, 1, 2, 0.4], [1, 0, 2, 1, 1, 1, 0.4], [1, 0, 2, 1, 1, 0, 0.2],
            [0, 2, 2, 1, 1, 2, 0.4], [0, 2, 2, 1, 1, 1, 0.4], [0, 2, 2, 1, 1, 0, 0.2],
            [0, 1, 2, 1, 1, 2, 0.4], [0, 1, 2, 1, 1, 1, 0.4], [0, 1, 2, 1, 1, 0, 0.2],
            [0, 0, 2, 1, 1, 2, 0.4], [0, 0, 2, 1, 1, 1, 0.4], [0, 0, 2, 1, 1, 0, 0.2],
            [1, 2, 1, 1, 1, 2, 0.4], [1, 2, 1, 1, 1, 1, 0.4], [1, 2, 1, 1, 1, 0, 0.2],
            [1, 1, 1, 1, 1, 2, 0.4], [1, 1, 1, 1, 1, 1, 0.4], [1, 1, 1, 1, 1, 0, 0.2],
            [1, 0, 1, 1, 1, 2, 0.4], [1, 0, 1, 1, 1, 1, 0.4], [1, 0, 1, 1, 1, 0, 0.2],
            [0, 2, 1, 1, 1, 2, 0.4], [0, 2, 1, 1, 1, 1, 0.4], [0, 2, 1, 1, 1, 0, 0.2],
            [0, 1, 1, 1, 1, 2, 0.4], [0, 1, 1, 1, 1, 1, 0.4], [0, 1, 1, 1, 1, 0, 0.2],
            [0, 0, 1, 1, 1, 2, 0.4], [0, 0, 1, 1, 1, 1, 0.4], [0, 0, 1, 1, 1, 0, 0.2],
            [1, 2, 0, 1, 1, 2, 0.4], [1, 2, 0, 1, 1, 1, 0.4], [1, 2, 0, 1, 1, 0, 0.2],
            [1, 1, 0, 1, 1, 2, 0.4], [1, 1, 0, 1, 1, 1, 0.4], [1, 1, 0, 1, 1, 0, 0.2],
            [1, 0, 0, 1, 1, 2, 0.4], [1, 0, 0, 1, 1, 1, 0.4], [1, 0, 0, 1, 1, 0, 0.2],
            [0, 2, 0, 1, 1, 2, 0.4], [0, 2, 0, 1, 1, 1, 0.4], [0, 2, 0, 1, 1, 0, 0.2],
            [0, 1, 0, 1, 1, 2, 0.4], [0, 1, 0, 1, 1, 1, 0.4], [0, 1, 0, 1, 1, 0, 0.2],
            [0, 0, 0, 1, 1, 2, 0.4], [0, 0, 0, 1, 1, 1, 0.4], [0, 0, 0, 1, 1, 0, 0.2],
            [1, 2, 2, 0, 1, 2, 0.4], [1, 2, 2, 0, 1, 1, 0.4], [1, 2, 2, 0, 1, 0, 0.2],
            [1, 1, 2, 0, 1, 2, 0.4], [1, 1, 2, 0, 1, 1, 0.4], [1, 1, 2, 0, 1, 0, 0.2],
            [1, 0, 2, 0, 1, 2, 0.4], [1, 0, 2, 0, 1, 1, 0.4], [1, 0, 2, 0, 1, 0, 0.2],
            [0, 2, 2, 0, 1, 2, 0.4], [0, 2, 2, 0, 1, 1, 0.4], [0, 2, 2, 0, 1, 0, 0.2],
            [0, 1, 2, 0, 1, 2, 0.4], [0, 1, 2, 0, 1, 1, 0.4], [0, 1, 2, 0, 1, 0, 0.2],
            [0, 0, 2, 0, 1, 2, 0.4], [0, 0, 2, 0, 1, 1, 0.4], [0, 0, 2, 0, 1, 0, 0.2],
            [1, 2, 1, 0, 1, 2, 0.4], [1, 2, 1, 0, 1, 1, 0.4], [1, 2, 1, 0, 1, 0, 0.2],
            [1, 1, 1, 0, 1, 2, 0.4], [1, 1, 1, 0, 1, 1, 0.4], [1, 1, 1, 0, 1, 0, 0.2],
            [1, 0, 1, 0, 1, 2, 0.4], [1, 0, 1, 0, 1, 1, 0.4], [1, 0, 1, 0, 1, 0, 0.2],
            [0, 2, 1, 0, 1, 2, 0.4], [0, 2, 1, 0, 1, 1, 0.4], [0, 2, 1, 0, 1, 0, 0.2],
            [0, 1, 1, 0, 1, 2, 0.4], [0, 1, 1, 0, 1, 1, 0.4], [0, 1, 1, 0, 1, 0, 0.2],
            [0, 0, 1, 0, 1, 2, 0.4], [0, 0, 1, 0, 1, 1, 0.4], [0, 0, 1, 0, 1, 0, 0.2],
            [1, 2, 0, 0, 1, 2, 0.4], [1, 2, 0, 0, 1, 1, 0.4], [1, 2, 0, 0, 1, 0, 0.2],
            [1, 1, 0, 0, 1, 2, 0.4], [1, 1, 0, 0, 1, 1, 0.4], [1, 1, 0, 0, 1, 0, 0.2],
            [1, 0, 0, 0, 1, 2, 0.4], [1, 0, 0, 0, 1, 1, 0.4], [1, 0, 0, 0, 1, 0, 0.2],
            [0, 2, 0, 0, 1, 2, 0.4], [0, 2, 0, 0, 1, 1, 0.4], [0, 2, 0, 0, 1, 0, 0.2],
            [0, 1, 0, 0, 1, 2, 0.4], [0, 1, 0, 0, 1, 1, 0.4], [0, 1, 0, 0, 1, 0, 0.2],
            [0, 0, 0, 0, 1, 2, 0.4], [0, 0, 0, 0, 1, 1, 0.4], [0, 0, 0, 0, 1, 0, 0.2],
            [1, 2, 2, 1, 0, 2, 0.4], [1, 2, 2, 1, 0, 1, 0.4], [1, 2, 2, 1, 0, 0, 0.2],
            [1, 1, 2, 1, 0, 2, 0.4], [1, 1, 2, 1, 0, 1, 0.4], [1, 1, 2, 1, 0, 0, 0.2],
            [1, 0, 2, 1, 0, 2, 0.4], [1, 0, 2, 1, 0, 1, 0.4], [1, 0, 2, 1, 0, 0, 0.2],
            [0, 2, 2, 1, 0, 2, 0.4], [0, 2, 2, 1, 0, 1, 0.4], [0, 2, 2, 1, 0, 0, 0.2],
            [0, 1, 2, 1, 0, 2, 0.4], [0, 1, 2, 1, 0, 1, 0.4], [0, 1, 2, 1, 0, 0, 0.2],
            [0, 0, 2, 1, 0, 2, 0.4], [0, 0, 2, 1, 0, 1, 0.4], [0, 0, 2, 1, 0, 0, 0.2],
            [1, 2, 1, 1, 0, 2, 0.4], [1, 2, 1, 1, 0, 1, 0.4], [1, 2, 1, 1, 0, 0, 0.2],
            [1, 1, 1, 1, 0, 2, 0.4], [1, 1, 1, 1, 0, 1, 0.4], [1, 1, 1, 1, 0, 0, 0.2],
            [1, 0, 1, 1, 0, 2, 0.4], [1, 0, 1, 1, 0, 1, 0.4], [1, 0, 1, 1, 0, 0, 0.2],
            [0, 2, 1, 1, 0, 2, 0.4], [0, 2, 1, 1, 0, 1, 0.4], [0, 2, 1, 1, 0, 0, 0.2],
            [0, 1, 1, 1, 0, 2, 0.4], [0, 1, 1, 1, 0, 1, 0.4], [0, 1, 1, 1, 0, 0, 0.2],
            [0, 0, 1, 1, 0, 2, 0.4], [0, 0, 1, 1, 0, 1, 0.4], [0, 0, 1, 1, 0, 0, 0.2],
            [1, 2, 0, 1, 0, 2, 0.4], [1, 2, 0, 1, 0, 1, 0.4], [1, 2, 0, 1, 0, 0, 0.2],
            [1, 1, 0, 1, 0, 2, 0.4], [1, 1, 0, 1, 0, 1, 0.4], [1, 1, 0, 1, 0, 0, 0.2],
            [1, 0, 0, 1, 0, 2, 0.4], [1, 0, 0, 1, 0, 1, 0.4], [1, 0, 0, 1, 0, 0, 0.2],
            [0, 2, 0, 1, 0, 2, 0.4], [0, 2, 0, 1, 0, 1, 0.4], [0, 2, 0, 1, 0, 0, 0.2],
            [0, 1, 0, 1, 0, 2, 0.4], [0, 1, 0, 1, 0, 1, 0.4], [0, 1, 0, 1, 0, 0, 0.2],
            [0, 0, 0, 1, 0, 2, 0.4], [0, 0, 0, 1, 0, 1, 0.4], [0, 0, 0, 1, 0, 0, 0.2],
            [1, 2, 2, 0, 0, 2, 0.4], [1, 2, 2, 0, 0, 1, 0.4], [1, 2, 2, 0, 0, 0, 0.2],
            [1, 1, 2, 0, 0, 2, 0.4], [1, 1, 2, 0, 0, 1, 0.4], [1, 1, 2, 0, 0, 0, 0.2],
            [1, 0, 2, 0, 0, 2, 0.4], [1, 0, 2, 0, 0, 1, 0.4], [1, 0, 2, 0, 0, 0, 0.2],
            [0, 2, 2, 0, 0, 2, 0.4], [0, 2, 2, 0, 0, 1, 0.4], [0, 2, 2, 0, 0, 0, 0.2],
            [0, 1, 2, 0, 0, 2, 0.4], [0, 1, 2, 0, 0, 1, 0.4], [0, 1, 2, 0, 0, 0, 0.2],
            [0, 0, 2, 0, 0, 2, 0.4], [0, 0, 2, 0, 0, 1, 0.4], [0, 0, 2, 0, 0, 0, 0.2],
            [1, 2, 1, 0, 0, 2, 0.4], [1, 2, 1, 0, 0, 1, 0.4], [1, 2, 1, 0, 0, 0, 0.2],
            [1, 1, 1, 0, 0, 2, 0.4], [1, 1, 1, 0, 0, 1, 0.4], [1, 1, 1, 0, 0, 0, 0.2],
            [1, 0, 1, 0, 0, 2, 0.4], [1, 0, 1, 0, 0, 1, 0.4], [1, 0, 1, 0, 0, 0, 0.2],
            [0, 2, 1, 0, 0, 2, 0.4], [0, 2, 1, 0, 0, 1, 0.4], [0, 2, 1, 0, 0, 0, 0.2],
            [0, 1, 1, 0, 0, 2, 0.4], [0, 1, 1, 0, 0, 1, 0.4], [0, 1, 1, 0, 0, 0, 0.2],
            [0, 0, 1, 0, 0, 2, 0.4], [0, 0, 1, 0, 0, 1, 0.4], [0, 0, 1, 0, 0, 0, 0.2],
            [1, 2, 0, 0, 0, 2, 0.4], [1, 2, 0, 0, 0, 1, 0.4], [1, 2, 0, 0, 0, 0, 0.2],
            [1, 1, 0, 0, 0, 2, 0.4], [1, 1, 0, 0, 0, 1, 0.4], [1, 1, 0, 0, 0, 0, 0.2],
            [1, 0, 0, 0, 0, 2, 0.4], [1, 0, 0, 0, 0, 1, 0.4], [1, 0, 0, 0, 0, 0, 0.2],
            [0, 2, 0, 0, 0, 2, 0.4], [0, 2, 0, 0, 0, 1, 0.4], [0, 2, 0, 0, 0, 0, 0.2],
            [0, 1, 0, 0, 0, 2, 0.4], [0, 1, 0, 0, 0, 1, 0.4], [0, 1, 0, 0, 0, 0, 0.2],
            [0, 0, 0, 0, 0, 2, 0.4], [0, 0, 0, 0, 0, 1, 0.4], [0, 0, 0, 0, 0, 0, 0.2]],
            [Laktat, BE, Bicarbonat, paCO2, ards])

        # Initialization of the nodes
        # Nodes that are commented out are not used for this Bayesian network
        ards1 = Node(ards, name="ARDS")
        compliance2 = Node(Compliance, name="Compliance")
        af4 = Node(Atemfrequenz, name="Atemfrequenz")
        be5 = Node(BE, name="BE")
        albumin6 = Node(Albunin, name="Albumin")
        bilirubin7 = Node(Bilirubin, name="Bilirubin")
        bnp8 = Node(BNP, name="BNP")
        ck9 = Node(CK, name="CK")
        ckmb10 = Node(CKMB, name="CK-MB")
        haemoglobin11 = Node(Hamoglobin, name="Hämoglobin")
        hf12 = Node(HF, name="Herzfrequenz")
        temp13 = Node(Korperkerntemperatur, name="Körperkerntemperatur")
        kreatinin14 = Node(Kreatinin, name="Kreatinin")
        pei15 = Node(PEI, name="P EI")
        paco216 = Node(paCO2, name="paCO2")
        pao217 = Node(paO2, name="paO2")
        peep19 = Node(PEEP, name="PEEP")
        ph20 = Node(ph, name="ph")
        spo222 = Node(SpO2, name="SpO2")
        thrombozyten23 = Node(Thrombozyten, name="Thrombozyten")
        troponin24 = Node(Troponin, name="Troponin")
        bicarbonat26 = Node(Bicarbonat, name="Bicarbonat")
        lactat27 = Node(Laktat, name="Laktat")
        leukozyten28 = Node(Leukozyten, name="Leukozyten")
        h2430 = Node(h24Bilanz, name="24h-Bilanz")
        deltap34 = Node(deltaP, name="delta P")
        got37 = Node(GOT, name="GOT")
        gpt38 = Node(GPT, name="GPT")
        harnstoff39 = Node(Harnstoff, name="Harnstoff")
        herzversagen40 = Node(Herzversagen, name="Herzversagen")
        horovitz41 = Node(Horovitzquotient, name="Horovitzquotient")
        hypervolamie42 = Node(Hypervolamie, name="Hypervolämie")
        indvt44 = Node(individuellesVT, name="individuelles VT pro kg Körpergewicht")
        lungenodem46 = Node(Lungenodem, name="Lungenödem")
        lymphozyten47 = Node(Lymphozyten, name="Lymphozyten")
        mb48 = Node(MaschinelleBeatmung, name="maschinelle Beatmung")
        ntprobnp49 = Node(NTproBNP, name="NT-pro BNP")
        pneumonie51 = Node(Pneumonie, name="Pneumonie")
        rontgen52 = Node(Rontgenbild, name="Röntgenbild")
        sepsis53 = Node(Sepsis, name="Sepsis")
        thoraxtrauma54 = Node(Thoraxtrauma, name="Thoraxtrauma")

        # Initialisation of the BN
        model = BayesianNetwork("ARDSBN")

        # adding the nodes to the BN
        # this is the ordering the data for the parameter learning has to be in
        model.add_states(ards1, compliance2, af4, be5, albumin6, bilirubin7, bnp8, ck9, ckmb10, haemoglobin11, hf12,
                        temp13, kreatinin14,
                        pei15, paco216, pao217, peep19, ph20, spo222, thrombozyten23, troponin24,
                        bicarbonat26, lactat27,
                        leukozyten28, h2430, deltap34, got37, gpt38, harnstoff39,
                        herzversagen40,
                        horovitz41, hypervolamie42, indvt44, lungenodem46, lymphozyten47, mb48,
                        ntprobnp49,
                        pneumonie51, rontgen52, sepsis53, thoraxtrauma54)

        # Adding all edges
        # edges that are commented out are not used for this Bayesian network
        model.add_edge(sepsis53, ards1)
        model.add_edge(pneumonie51, ards1)
        model.add_edge(thoraxtrauma54, ards1)

        model.add_edge(ards1, kreatinin14)
        model.add_edge(ards1, harnstoff39)
        model.add_edge(ards1, rontgen52)
        model.add_edge(ards1, hf12)
        model.add_edge(ards1, ph20)
        model.add_edge(ards1, haemoglobin11)

        model.add_edge(ards1, paco216)

        model.add_edge(ards1, spo222)
        model.add_edge(ards1, pao217)

        model.add_edge(ards1, mb48)
        model.add_edge(ards1, compliance2)
        model.add_edge(mb48, compliance2)

        model.add_edge(mb48, indvt44)
        model.add_edge(mb48, pei15)
        model.add_edge(mb48, peep19)
        model.add_edge(mb48, horovitz41)
        model.add_edge(mb48, af4)
        model.add_edge(ards1, horovitz41)
        model.add_edge(peep19, deltap34)
        model.add_edge(pei15, deltap34)
        model.add_edge(paco216, ph20)
        model.add_edge(lactat27, ph20)
        model.add_edge(be5, ph20)
        model.add_edge(bicarbonat26, ph20)
        model.add_edge(sepsis53, kreatinin14)
        model.add_edge(sepsis53, harnstoff39)
        model.add_edge(sepsis53, hypervolamie42)

        model.add_edge(sepsis53, temp13)
        model.add_edge(sepsis53, leukozyten28)
        model.add_edge(sepsis53, gpt38)
        model.add_edge(sepsis53, albumin6)
        model.add_edge(sepsis53, got37)
        model.add_edge(sepsis53, thrombozyten23)
        model.add_edge(sepsis53, bilirubin7)

        model.add_edge(leukozyten28, lymphozyten47)
        model.add_edge(h2430, hypervolamie42)
        model.add_edge(herzversagen40, ck9)

        model.add_edge(herzversagen40, lungenodem46)
        model.add_edge(herzversagen40, bnp8)
        model.add_edge(herzversagen40, troponin24)
        model.add_edge(herzversagen40, hf12)

        model.add_edge(bnp8, ntprobnp49)
        model.add_edge(ck9, ckmb10)
        model.add_edge(hypervolamie42, lungenodem46)
        model.add_edge(lungenodem46, rontgen52)

        # filalize the topology of the model
        model.bake()

        return model

