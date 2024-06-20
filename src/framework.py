from processing import *
from models import *
from metrics import *
from evaluation import *

from visualization import plot_eval
from cli import make_parser

import os
import yaml
from datetime import datetime
import json


class Framework:
    def __init__(self):
        args = make_parser().parse_args()
        if args.config:
            config = json.loads(args.config)
        else:
            with open(args.config_file, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config
        self.loader = FileLoader()
        self.timeseries_models = []
        self.timeseries_classes = []
        for model in config["timeseries_models_to_execute"]:
            self.timeseries_models.append(eval(model + "()"))
            self.timeseries_classes.append(eval(model))
        self.evaluator = Evaluation(self.timeseries_classes, config["evaluation"])
        self.image_dl_methods = ["VIT"]
        self.image_models = [VisionTransformer(config["image_model_parameters"], "vit-small-16")]
        self.timeseries_training_data = None
        self.timeseries_test_data = None
        self.image_pneumonia_training_data = None
        self.image_ards_training_data = None
        self.image_ards_test_data = None
        self.pneumonia_dataset = config["data"]["pneumonia_dataset"]
        self.ards_dataset = config["data"]["ards_dataset"]
        self.dataProcessor = DataProcessor(config["processing"], config["data"]["database"], config["process"])
        self.feature_selector = Feature_selection(config["feature_selection"])
        self.segregator = Data_segregator(config["data_segregation"])
        self.dataset_generator = DatasetGenerator()
        self.process = config["process"]
        self.loading_paths = config["loading_paths"]
        self.timeseries_file_path = config["data"]["timeseries_file_path"]
        self.image_file_path = config["data"]["image_file_path"]
        self.method = config["image_model_parameters"]["method"]
        self.mode = config["image_model_parameters"]["mode"]
        self.outdir = "./Save/" + str(datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/"
        if not self.outdir.endswith("/"):
            self.outdir += "/"

    def load_timeseries_data(self):
        dataframe = self.loader.load_file(self.timeseries_file_path)
        print(dataframe)
        dataframe = self.dataProcessor.process_data(dataframe)
        if self.process["perform_feature_selection"] == True:
            dataframe = self.feature_selector.perform_feature_selection(dataframe)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        dataframe.to_csv(self.outdir + os.path.basename(self.timeseries_file_path) + "_preprocessed.csv", index=True)
        print("Finished preprocessing and saved result to file!")
        if self.process["perform_data_segregation"] == True:
            training_data, test_data = self.segregator.segregate_data(dataframe)
            self.timeseries_training_data = training_data
            self.timeseries_test_data = test_data
        else:
            self.timeseries_test_data = self.timeseries_training_data = dataframe

        self.timeseries_test_data.to_csv(self.outdir + "test_data.csv", header=True, index=False)
        self.timeseries_training_data.to_csv(self.outdir + "training_data.csv", header=True, index=False)

    def load_image_data(self):
        for dl_method in self.image_dl_methods:
            self.image_pneumonia_training_data = self.dataset_generator.build_dataset(self.pneumonia_dataset, dl_method,
                                                                                      'PNEUMONIA',
                                                                                      path=self.image_file_path,
                                                                                      augment=False)
            self.image_ards_training_data = self.dataset_generator.build_dataset(self.ards_dataset, dl_method, 'ARDS',
                                                                                 path=self.image_file_path,
                                                                                 augment=False)
            self.image_ards_test_data = self.dataset_generator.build_dataset('test', dl_method, 'ARDS',
                                                                             path=self.image_file_path, augment=False)

    def learn_timeseries_models(self):
        for model in self.timeseries_models:
            model.train_model(self.timeseries_training_data)
            print("Successfully trained " + model.name)

    def learn_image_models(self):
        for model in self.image_models:
            info_list = [self.pneumonia_dataset, self.ards_dataset, model.model_name, [self.method, self.method],
                         self.mode]
            model.train_image_model(self.image_pneumonia_training_data, self.image_ards_training_data, info_list)

    def test_image_models(self):
        for model in self.image_models:
            info_list = [self.pneumonia_dataset, self.ards_dataset, model.model_name, self.method, self.mode]
            model.test_image_model(self.image_ards_test_data, info_list)

    def predict(self, test_data):
        print("------------")
        print(test_data)
        print("------------")
        test_data = test_data.drop(columns=['ards'])
        for model in self.timeseries_models:
            prediction = model.predict(test_data)
            print("Classification of " + model.name + ": ")
            print(prediction)

    def evaluate_models(self):
        result = {}
        if self.process["calculate_evaluation_metrics"] == True:
            for model in self.timeseries_models:
                # for each model, add corresponding dict to results dict
                result[model.name] = self.evaluator.evaluate(model, self.timeseries_test_data)

        if self.process["perform_cross_validation"] == True:
            cross_validation_results = self.evaluator.perform_cross_validation(self.timeseries_test_data, self.outdir)
            for model_name in list(cross_validation_results.keys()):
                # for each model, add corresponding cross validation results to already existing
                # data in json dict
                result[model_name]["cross_validation"] = cross_validation_results[model_name]
        if result:
            print(f"Save results to {self.outdir + 'results.json'}")
            with (open(self.outdir + 'results.json', 'w', encoding='utf-8') as f):
                json.dump(result, f, ensure_ascii=False, indent=4)
                plot_eval(data=result, file_name=f.name)

    def save_models(self):
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        for model in self.timeseries_models:
            model.save(self.outdir + model.name)
            #file.write(str(model_serialized))
        print("Successfully stored all models!")

    def load_models(self):
        for model in self.timeseries_models:
            if model.name in self.loading_paths:
                if self.loading_paths[model.name] != "default":
                    model.load(self.loading_paths[model.name])
                else:
                    model.load(self.outdir + model.name)
            else:
                model.load(self.outdir + model.name)
        print("Successfully loaded all models!")

    def execute(self):
        # Store configuration in outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        with open(self.outdir + 'config.json', 'w') as f:
            json.dump(self.config, f)

        if self.process["load_models"] == True:
            self.load_models()

        if self.process["load_timeseries_data"] == True:
            self.load_timeseries_data()

        if self.process["perform_timeseries_training"] == True:
            self.learn_timeseries_models()

        if self.process["perform_timeseries_classification"] == True:
            self.predict(self.timeseries_test_data)

        self.evaluate_models()

        if self.process["save_models"] == True:
            self.save_models()

        if self.process["load_image_data"] == True:
            self.load_image_data()

        if self.process["train_image_models"] == True:
            self.learn_image_models()

        if self.process["test_image_models"] == True:
            self.test_image_models()
