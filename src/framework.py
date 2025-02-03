from cli import make_parser
from processing import DataFileManager, DataProcessor, FeatureSelection, DataSegregator, ImageDatasetGenerator, TimeSeriesDatasetManagement, TimeSeriesDataset
from ml_models import *
from evaluation import Evaluation
from metrics import ResultManagement

import os
import yaml
import json
import pandas as pd

from datetime import datetime
from pathlib import Path




class Framework:
    def __init__(self):
        args = make_parser().parse_args()
        if args.config:
            config = json.loads(args.config)
        else:
            with open(args.config_file, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config
        self.loader = DataFileManager()
        self.supported_timeseries_models = self.config['supported_algorithms']['timeseries_models']
        self.supported_images_models = self.config['supported_algorithms']['image_models']

        self.available_timeseries_models = {}

        for model in self.supported_timeseries_models:
            algorithm = model.replace("Model", "")
            self.available_timeseries_models[algorithm] = []

        self.timeseries_model_use_config = self.config['models']['timeseries_models']

        self.timeseries_models_to_train = self.create_needed_models(self.timeseries_model_use_config, 'to_train')
        self.timeseries_models_to_evaluate = self.create_needed_models(self.timeseries_model_use_config, 'to_evaluate')
        self.timeseries_models_to_execute = self.create_needed_models(self.timeseries_model_use_config, 'to_execute')
        self.timeseries_models_to_cross_validate = self.create_needed_models(self.timeseries_model_use_config, 'to_cross_validate')

        self.timeseries_training_set = None
        self.timeseries_test_set = None
        self.timeseries_complete_set = None

        self.timeseries_evaluations_result = None
        self.timeseries_cross_validation_result = None
        self.processing_meta_data = {}

        self.image_dl_methods = ["VIT"]
        self.image_models = [VisionTransformerModel(config["image_model_parameters"], "vit-small-16")]

        self.image_pneumonia_training_data = None
        self.image_ards_training_data = None
        self.image_ards_test_data = None
        self.pneumonia_dataset = config["data"]["pneumonia_dataset"]
        self.ards_dataset = config["data"]["ards_dataset"]
        self.dataProcessor = DataProcessor(config["preprocessing"], config["data"]["database"], config["process"])
        self.feature_selector = FeatureSelection(config["feature_selection"])
        self.segregator = DataSegregator(config["data_segregation"])
        self.dataset_generator = ImageDatasetGenerator()
        self.process = config["process"]
        self.model_base_paths = config["algorithm_base_path"]
        self.timeseries_file_path = config["data"]["timeseries_file_path"]
        self.image_file_path = config["data"]["image_file_path"]
        self.method = config["image_model_parameters"]["method"]
        self.mode = config["image_model_parameters"]["mode"]
        self.outdir = config["storage_path"] if config["storage_path"] else "./Save/" + str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "/"
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if not self.outdir.endswith("/"):
            self.outdir += "/"

        self.TimeseriesModelManager = TimeSeriesModelManager(config, self.outdir)

    @staticmethod
    def create_needed_models(config: dict, stage: str) -> dict[str, list[str]]:
        stage_config = config[stage]
        result_dict = {}
        for model_algorithm in stage_config.keys():

            if stage_config[model_algorithm]["Active"]:
                content = stage_config[model_algorithm]
                del content["Active"]
                result_dict[model_algorithm] = content
        return result_dict


    def load_timeseries_data(self):
        dataframe, dataset_metadata = self.loader.load_file(self.timeseries_file_path)

        if dataset_metadata:
            self.dataProcessor.database_name = dataset_metadata.datasource

        dataframe = self.dataProcessor.process_data(dataframe, dataset_metadata)
        processing_meta_data = self.dataProcessor.get_processing_meta_data()

        if self.process["perform_feature_selection"]:
            dataframe = self.feature_selector.perform_feature_selection(dataframe)
            self.feature_selector.create_meta_data()
        processing_meta_data["feature_selection"] = self.feature_selector.meta_data

        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

        print("Finished preprocessing and saved result to file!")
        if self.process["perform_data_segregation"]:
            training_data, test_data = self.segregator.segregate_data(dataframe)
        else:
            print("Warning: Training and Test data are the same!")
            training_data = test_data = dataframe


        current_time_str = str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        base_path_data = self.outdir + os.path.basename(self.timeseries_file_path) + f"_{current_time_str}_"
        path_complete = base_path_data + "complete"
        path_training = base_path_data + "training"
        path_test = base_path_data + "test"

        self.timeseries_training_set = TimeSeriesDatasetManagement.factory_method(training_data, processing_meta_data,path_training, "Training")
        self.timeseries_test_set = TimeSeriesDatasetManagement.factory_method(test_data, processing_meta_data, path_test, "Test")
        self.timeseries_complete_set = TimeSeriesDatasetManagement.factory_method(test_data, processing_meta_data,path_complete, "Complete")
        TimeSeriesDatasetManagement.write(self.timeseries_complete_set)
        TimeSeriesDatasetManagement.write(self.timeseries_training_set)
        TimeSeriesDatasetManagement.write(self.timeseries_test_set)

    def load_timeseries_models(self, stage: str):

        if stage == "to_execute":
            needed_models = self.timeseries_models_to_execute
        elif stage == "to_evaluate":
            needed_models = self.timeseries_models_to_evaluate
        else:
            print("For stage " + stage + " it does not make sense to load models")
            return
        self.available_timeseries_models = self.TimeseriesModelManager.load_models(needed_models, self.available_timeseries_models, self.model_base_paths)


    def learn_timeseries_models(self):

        model_dict = self.TimeseriesModelManager.create_model_from_config(self.timeseries_models_to_train,
                                                                          self.timeseries_model_use_config["base_path_config"]["to_train"])

        for model_type, models in model_dict.items():
            for model in models:
                model.train_timeseries(self.timeseries_training_set, self.config, "Training")
                if self.process["save_models"]:
                    if self.config["algorithm_base_path"][model.algorithm] != "default":
                        save_path = self.config["algorithm_base_path"][model.algorithm]
                    else:
                        save_path = self.outdir
                    model.save(save_path)
                print("Successfully trained " + model.name + " for algorithm " + model.algorithm)
                self.available_timeseries_models[model.algorithm].append(model)


    def execute_timeseries_models(self, test_set: TimeSeriesDataset):
        self.load_timeseries_models("to_execute")
        test_data = test_set.content
        input_features = test_data.drop(columns=['ards'])
        test_data = test_data.rename(columns={"ards": "ards_diagnosed"}).reset_index(drop=True)



        for model_type, model_names in self.timeseries_models_to_execute.items():
            for model in self.available_timeseries_models[model_type]:
                if model.name not in model_names["Names"]:
                    continue
                prediction = model.predict(input_features)
                print(f"Finished prediction of {model.name}")
                df = pd.DataFrame({"ards_predicted": prediction}).reset_index(drop=True)
                df = pd.concat([test_data, df], axis=1)
                df.to_csv(self.outdir + f"prediction_{model.algorithm}_{model.name}.csv", index=False)

    def evaluate_timeseries_models(self):
        self.load_timeseries_models("to_evaluate")

        models_to_evaluate_dict = {}
        for model_type, model_name in self.timeseries_models_to_evaluate.items():
            for model in self.available_timeseries_models[model_type]:
                if model.name not in model_name["Names"]:
                    continue
                if model_type in models_to_evaluate_dict:
                    models_to_evaluate_dict[model_type].append(model)
                else:
                    models_to_evaluate_dict[model_type] = [model]

        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)
        overall_result = evaluator.evaluate_timeseries_models(models_to_evaluate_dict)
        self.timeseries_evaluations_result = overall_result

    def cross_validate_models(self):
        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)

        models_to_cross_validate_dict =  self.TimeseriesModelManager.create_model_from_config(
            self.timeseries_models_to_cross_validate, self.timeseries_model_use_config["base_path_config"]["to_cross_validate"])

        overall_result = evaluator.cross_validate_timeseries_models(models_to_cross_validate_dict)
        self.timeseries_cross_validation_result = overall_result


    def handle_timeseries_results(self):

        # TODO make plots
        eval_name = self.config['evaluation']['evaluation_name']
        eval_file_name = eval_name.replace(" ", "_") + "_results.json"
        result_location = self.outdir + eval_file_name

        if self.timeseries_cross_validation_result and self.timeseries_evaluations_result:

            cv_result = self.timeseries_cross_validation_result
            eval_result = self.timeseries_evaluations_result

            content = {"eval_name": eval_name, "storage_location": result_location, "CV": cv_result, "Eval": eval_result}
            final_result = ResultManagement().merge(content, "CV_EVAL")

        elif self.timeseries_cross_validation_result:
            final_result = self.timeseries_cross_validation_result

        elif self.timeseries_evaluations_result:
            final_result = self.timeseries_evaluations_result

        else:
            print("This should never happen")
            return

        print(f"Save results to {self.outdir + 'results.json'}")
        with open(result_location, 'w', encoding='utf-8') as f:
            f.write(final_result.model_dump_json(indent=4))

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

    def load_image_models(self):
        raise NotImplementedError
        pass

    def learn_image_models(self):
        for model in self.image_models:
            info_list = [self.pneumonia_dataset, self.ards_dataset, model.model_name, [self.method, self.method],
                         self.mode]
            model.train_image_model(self.image_pneumonia_training_data, self.image_ards_training_data, info_list)

    def evaluate_image_models(self):
        for model in self.image_models:
            info_list = [self.pneumonia_dataset, self.ards_dataset, model.model_name, self.method, self.mode]
            model.test_image_model(self.image_ards_test_data, info_list)


    def run(self):
        # Store configuration in outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        with open(self.outdir + 'config.json', 'w') as f:
            json.dump(self.config, f)


        if self.process["load_timeseries_data"]:
            self.load_timeseries_data()


        if self.process["perform_timeseries_training"]:
            if not self.timeseries_training_set:
                print("Can not train without training data. Exiting...")
                exit()
            self.learn_timeseries_models()

        if self.process["perform_timeseries_classification"]:
            if not self.timeseries_test_set:
                print("Can not predict without test data. Exiting...")
                exit()
            self.execute_timeseries_models(self.timeseries_test_set)

        if self.process["calculate_evaluation_metrics"]:
            self.evaluate_timeseries_models()

        if self.process["perform_cross_validation"]:
            if not self.timeseries_training_set:
                print("Can not cross validate without training data. Exiting...")
                exit()
            self.cross_validate_models()

        if self.timeseries_evaluations_result or self.timeseries_cross_validation_result:

            self.handle_timeseries_results()


        if self.process["load_image_data"]:
            self.load_image_data()

        if self.process["train_image_models"]:
            self.learn_image_models()

        if self.process["test_image_models"]:
            self.evaluate_image_models()
