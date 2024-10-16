import fnmatch

from processing import *
from ml_models.vision_transformer import VisionTransformerModel
from evaluation import ModelEvaluation, Evaluation
from ml_models import *

from visualization import plot_eval
from metrics.Models import ModelResult, EvalResult, ExperimentResult
from cli import make_parser

import os
import yaml
from datetime import datetime
import json
from pathlib import Path

# TODO make sure we have trained models for evaluation



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
        self.supported_timeseries_models = self.config['supported_algorithms']['timeseries_models']
        self.supported_images_models = self.config['supported_algorithms']['image_models']
        self.timeseries_models_to_train = []
        self.timeseries_models_to_evaluate = []
        self.timeseries_algorithms_to_evaluate = []
        

        for model in config['timeseries_models_to_train']:
            self.timeseries_models_to_train.append(str(model))

        for model in config['timeseries_models_to_evaluate']:
            self.timeseries_models_to_evaluate.append(str(model[1]))
            self.timeseries_algorithms_to_evaluate.append(str(model[0]))

        


        self.trained_timeseries_models  = []

        self.timeseries_models = []
        self.timeseries_classes = []
        for model in config["timeseries_models_to_execute"]:
            self.timeseries_models.append(eval(model + "()"))
            self.timeseries_classes.append(eval(model))
        #self.evaluator = Evaluation_Old(self.timeseries_classes, config["evaluation"])
        self.image_dl_methods = ["VIT"]
        self.image_models = [VisionTransformerModel(config["image_model_parameters"], "vit-small-16")]
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
        self.dataset_generator = ImageDatasetGenerator()
        self.process = config["process"]
        self.model_base_paths = config["algorithm_base_path"]
        self.timeseries_file_path = config["data"]["timeseries_file_path"]
        self.image_file_path = config["data"]["image_file_path"]
        self.method = config["image_model_parameters"]["method"]
        self.mode = config["image_model_parameters"]["mode"]
        self.outdir = config["storage_path"] if config["storage_path"] else "./Save/" + str(
            datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/"
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
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
        if self.process["perform_data_segregation"]:
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
            model.train_timeseries(self.timeseries_training_data, self.config)
            if self.process["save_models"]:
                training_data_location = ""
                model.save(self.outdir + model.name, training_data_location)



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

    #def evaluate_timeseries_models(self):


    def evaluate_models(self):
        result = {}

        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_data,
                               dataset_test=self.timeseries_test_data)
        overall_result = evaluator.evaluate_timeseries_models(self.timeseries_models)

        if overall_result.contained_model_results:
            print(f"Save results to {self.outdir + 'results.json'}")
            with (open(self.outdir + 'results.json', 'w', encoding='utf-8') as f):
                f.write(overall_result.model_dump_json())
                #json.dump(overall_result, f, ensure_ascii=False, indent=4)
                # TODO make plots
                # plot_eval(data=result, file_name=f.name)

    def cross_validate_models(self):
        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_data,
                               dataset_test=self.timeseries_test_data)
        overall_result = evaluator.evaluate_timeseries_models(self.timeseries_models,)

    def save_models(self):
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        for model in self.timeseries_models:
            model.save_model(self.outdir + model.name)
            #file.write(str(model_serialized))
        print("Successfully stored all models!")

    def load_timeseries_models(self):
        models_to_load = set(self.timeseries_models_to_evaluate).difference(set(self.timeseries_models))
        for model_name in models_to_load:
            index = self.timeseries_models.index(model_name)
            algorithm = self.timeseries_algorithms_to_evaluate[index]
            base_path = self.model_base_paths[algorithm]
            model_obj = eval(algorithm + "()")
            loaded_model = model_obj.load(base_path, model_name)
            self.timeseries_models.append(loaded_model)





    def load_image_models(self):

        pass




    def load_models(self):
        self.load_image_models()
        self.load_timeseries_models()

        for model in self.timeseries_models:
            if model in self.model_base_paths:
                if self.model_base_paths[model] != "default":
                    model.load(self.model_base_paths[model.name])
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

        if self.process["load_models"]:
            self.load_models()

        if self.process["load_timeseries_data"]:
            self.load_timeseries_data()

        if self.process["perform_timeseries_training"]:
            self.learn_timeseries_models()

        if self.process["perform_timeseries_classification"]:
            self.predict(self.timeseries_test_data)

        if self.process["calculate_evaluation_metrics"]:
            self.evaluate_models()

        if self.process["perform_cross_validation"]:
            self.cross_validate_models()

        if self.process["save_models"]:
            self.save_models()

        if self.process["load_image_data"]:
            self.load_image_data()

        if self.process["train_image_models"]:
            self.learn_image_models()

        if self.process["test_image_models"]:
            self.test_image_models()
