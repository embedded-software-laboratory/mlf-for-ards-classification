
import pandas as pd


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

        self.timeseries_training_set = None
        self.timeseries_test_set = None
        self.timeseries_complete_set = None



        self.timeseries_evaluations_result = None
        self.timeseries_cross_validation_result = None
        self.processing_meta_data = {}


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
        self.image_dl_methods = ["VIT"]
        self.image_models = [VisionTransformerModel(config["image_model_parameters"], "vit-small-16")]

        self.image_pneumonia_training_data = None
        self.image_ards_training_data = None
        self.image_ards_test_data = None
        self.pneumonia_dataset = config["data"]["pneumonia_dataset"]
        self.ards_dataset = config["data"]["ards_dataset"]
        self.dataProcessor = DataProcessor(config["preprocessing"], config["data"]["database"], config["process"])
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
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "/"
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if not self.outdir.endswith("/"):
            self.outdir += "/"

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

            model.train_timeseries(self.timeseries_training_set, self.config, "Training")
            if self.process["save_models"]:

                model.save(self.outdir + model.name)
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

    def predict_timeseries_models(self, test_set: TimeSeriesDataset):
        test_data = test_set.content
        input_features = test_data.drop(columns=['ards'])
        test_data = test_data.rename(columns={"ards": "ards_diagnosed"}).reset_index(drop=True)
        for model in self.timeseries_models:
            prediction = model.predict(input_features)
            print(f"Finished prediction of {model.name}")
            df = pd.DataFrame({"ards_predicted": prediction}).reset_index(drop=True)
            df = pd.concat([test_data, df], axis=1)
            df.to_csv(self.outdir + f"prediction_{model.name}.csv", index=False)

    def evaluate_timeseries_models(self):

        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)
        overall_result = evaluator.evaluate_timeseries_models(self.timeseries_models)
        self.timeseries_evaluations_result = overall_result


    def cross_validate_models(self):
        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)
        overall_result = evaluator.cross_validate_timeseries_models(self.timeseries_models)
        self.timeseries_cross_validation_result = overall_result


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

    def handle_timeseries_results(self):

        # TODO make plots

        result_location = self.outdir + 'results.json'
        if self.timeseries_cross_validation_result and self.timeseries_evaluations_result:
            eval_name = self.config['evaluation']['evaluation_name']

            cv_result = self.timeseries_cross_validation_result
            eval_result = self.timeseries_evaluations_result
            contained_models = set()
            model_eval_dict = {}
            model_name_dict = {}
            model_storage_dict = {}

            for model in cv_result.contained_model_results.keys():
                contained_models.add(model)
            for model in eval_result.contained_model_results.keys():
                contained_models.add(model)
            for model in list(contained_models):
                model_eval_dict[model] = {}

                if model in eval_result.contained_model_results:
                    for key in eval_result.contained_model_results[model].contained_evals.keys():
                        model_eval_dict[model][key] = eval_result.contained_model_results[model].contained_evals[key]

                    model_name_dict[model] = eval_result.contained_model_results[model].used_model_name
                    model_storage_dict[model] = eval_result.contained_model_results[model].used_model_location
                if model in cv_result.contained_model_results:
                    for key in cv_result.contained_model_results[model].contained_evals.keys():
                        model_eval_dict[model][key] = cv_result.contained_model_results[model].contained_evals[key]
                    if model not in eval_result.contained_model_results.keys():
                        model_name_dict[model] = cv_result.contained_model_results[model].used_model_name
                        model_storage_dict[model] = cv_result.contained_model_results[model].used_model_location
            model_result_dict = {}
            for model in model_eval_dict.keys():
                contained_evals = model_eval_dict[model]


                model_name = model_name_dict[model]
                model_storage = model_storage_dict[model]

                model_result = ModelResult( used_model_location= model_storage, used_model_name=model_name,
                                            contained_evals=contained_evals)
                model_result_dict[model] = model_result
            final_result = ExperimentResult(result_name=eval_name, storage_location =result_location,
                                            contained_model_results= model_result_dict)

        elif self.timeseries_cross_validation_result:
            final_result = self.timeseries_cross_validation_result

        elif self.timeseries_evaluations_result:
            final_result = self.timeseries_evaluations_result

        else:
            print("This should never happen")
            return

        print(f"Save results to {self.outdir + 'results.json'}")
        with (open(result_location, 'w', encoding='utf-8') as f):
            f.write(final_result.model_dump_json(indent=4))





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
            if not self.timeseries_training_set:
                print("Can not train without training data. Exiting...")
                exit()
            self.learn_timeseries_models()

        if self.process["perform_timeseries_classification"]:
            if not self.timeseries_test_set:
                print("Can not predict without test data. Exiting...")
                exit()
            self.predict_timeseries_models(self.timeseries_test_set)

        if self.process["calculate_evaluation_metrics"]:
            self.evaluate_timeseries_models()

        if self.process["perform_cross_validation"]:
            if not self.timeseries_training_set:
                print("Can not cross validate without training data. Exiting...")
                exit()
            self.cross_validate_models()

        if self.timeseries_evaluations_result or self.timeseries_cross_validation_result:
            self.handle_timeseries_results()


        if self.process["save_models"]:
            self.save_models()

        if self.process["load_image_data"]:
            self.load_image_data()

        if self.process["train_image_models"]:
            self.learn_image_models()

        if self.process["test_image_models"]:
            self.test_image_models()
