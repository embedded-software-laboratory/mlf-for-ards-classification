# Configuration File Guide

The configuration files in the folder "src/configs/" (e.g., "config.yml") control which steps the framework should execute and allow setting various parameters that influence the final result.

**Important:** The config files are located in the `src/configs/` folder. There are multiple example configs for different scenarios (e.g., `config_Dissertation_Test.yml`, `config_Dissertation_Calibration_50.yml`, etc.).

## process

This section defines which steps should be executed. Each item can be set to "true" (will be executed) or "false" (will not be executed).

1. **load_models**: Controls whether saved models should be loaded. If set to true, the paths specified under "algorithm_base_path" will be used for loading the models.
2. **load_timeseries_data**: Controls whether time series data should be loaded. If true, the file specified under "data" will be loaded.
3. **perform_anomaly_detection**: If true, anomaly detection will be performed on the loaded data. Settings for anomaly detection are explained under [anomaly_detection](#anomaly_detection).
4. **perform_imputation**: If true, missing values in the loaded data will be imputed. More detailed imputation settings can be configured under "preprocessing/imputation".
   Completely empty columns cannot be imputed and will be automatically removed by the framework.
   If some columns have no values at all, this may cause crashes later. Therefore, it's generally advisable to activate this point and impute all parameters.
5. **perform_unit_conversion**: If true, the framework will attempt to convert parameter units to those specified by the Catalog of Items. For this, the database must be specified under "data/database". The framework then assumes that each parameter comes in the unit specified in the relevance table for the respective database.
   Currently implemented conversions:
   1. Database eICU:
      - Hemoglobin: g/dL to mmol/L
      - Creatinine: mg/dL to µmol/L
      - Albumin: g/dL to µmol/L
      - CRP: mg/L to nmol/L
      - etCO2: mmHg to %
      - Bilirubin: mg/dL to µmol/L
   2. Database MIMIC-3:
      - Hemoglobin: g/dL to mmol/L
      - Urea: mg/dL to mmol/L
      - Creatinine: mg/dL to µmol/L
      - Albumin: g/dL to µmol/L
      - Bilirubin: mg/dL to µmol/L
      - CRP: mg/L to nmol/L
   3. Database MIMIC-4:
      - Hemoglobin: g/dL to mmol/L
6. **calculate_missing_params**: If true, the framework will attempt to calculate missing parameters from the available data. Under "preprocessing/params_to_calculate", you specify which parameters should be calculated.
7. **perform_ards_onset_detection**: If true, the framework will determine for each individual patient (identified via the "patient_id" column) when ARDS likely first occurred and return this time point or a certain time span around it. The exact rule for determining ARDS onset and which data should be returned can be set under "preprocessing/ards_onset_detection".
   For better comparability, the most suitable time point is also identified for non-ARDS patients.
8. **perform_filtering**: If activated, the framework will execute the filters configured under "preprocessing/filtering" to filter out patients who may have an incorrect ARDS label. More detailed explanations of available filters are described in the preprocessing section.
9. **perform_feature_selection**: To increase the speed of the training process, feature selection can be used to calculate which parameters have a particularly high or low influence on the reference parameter. Subsequently, parameters with low influence are removed. Feature selection settings are made under "feature_selection".
10. **perform_data_segregation**: This step splits the loaded data into a dataset for model training and a dataset for evaluation/prediction. Additional settings can be made under "data_segregation".
    This step only needs to be activated if models should be trained and then evaluated in the same run. If this step is deactivated, the complete loaded dataset will be used for both classification/evaluation and training; therefore, deactivating the split only makes sense if either classification/evaluation or model training is to be performed.
11. **perform_timeseries_training**: When activated, all time series models specified under "models/timeseries_models/to_train" will be trained with the loaded data. If evaluation should be performed in the same run, "perform_data_segregation" should also be activated.
12. **perform_timeseries_classification**: If activated, all available time series models will classify the loaded data and output 1 for "ARDS" or 0 for "Not ARDS" for each row.
13. **perform_threshold_optimization**: If activated and possible for the selected model, the optimal decision boundary will be determined according to different algorithms. Which algorithms are used is defined under evaluation/threshold_optimization_algorithms.
14. **calculate_evaluation_metrics**: If activated, the framework will use the loaded test data to have the models classify them and calculate how good the model results are. Various metrics will be calculated for this.
15. **perform_cross_validation**: If activated, the models will be cross-validated with the loaded data. Some parameters for this can be configured under "evaluation".
16. **save_models**: If true, all trained models will be saved in the "Save" folder.
17. **load_image_data**: Controls whether image data for training the X-ray image models should be loaded. If true, the images will be loaded from the file path specified under data/image_file_path.
18. **train_image_models**: If activated, the models for X-ray images specified under "models/image_models/to_train" will be trained. This includes training for pneumonia detection as well as transfer learning to ARDS.
19. **execute_image_models**: If activated, trained image models will execute classification on loaded image data.
20. **test_image_models**: If activated, the trained image data models will be evaluated with the loaded ARDS image dataset.

## supported_algorithms

This section defines the algorithms supported by the framework. This serves as a central list of available models.

### timeseries_models
List of supported time series models:
- AdaBoostModel
- LightGBMModel
- LogisticRegressionModel
- RandomForestModel
- SupportVectorMachineModel (can be commented out if not needed)
- XGBoostModel

### image_models
List of supported image models:
- ResNetModel
- DenseNetModel
- ViTModel (Vision Transformer)

## models

This section defines which models should be used in different steps of the framework.

### timeseries_models
This section contains the models to be used for time series classification.

#### base_path_config
Here you specify the base paths for model configuration files:
- **to_cross_validate**: Path to config files for cross-validation
- **to_train**: Path to config files for training

For each phase of the framework (Training, Classification, Evaluation, Cross Validation), there is a separate list where different algorithms are specified. Individual algorithms must be activated by setting "Active: true".

The models to be used for individual phases are listed by name under "Names:". For the phases to_train and to_cross_validate, it is also possible to specify the names of configuration files for hyperparameters. These files must be located in a folder with the algorithm name in the directory specified under "base_path_config" (see example structure in Data). If no configuration file is specified, the value "default" must be given for this model.

**Phases:**
- **to_train**: Models to be trained (requires: Active, Configs, Names)
- **to_cross_validate**: Models for cross-validation (requires: Active, Configs, Names)
- **to_evaluate**: Models to be evaluated (requires: Active, Names)
- **to_execute**: Models to be executed for classification (requires: Active, Names)

The framework ensures that models not yet available through training are loaded from disk in the phase where they are needed. Loading from disk only makes sense for the phases to_execute and to_evaluate. The storage location is composed of the path specified under "algorithm_base_path" and the model name.

### image_models

Similar to timeseries_models, there are configurations for image models here. The same phases (to_train, to_cross_validate, to_evaluate, to_execute) are available.

For each model (ResNet, DenseNet, ViT):
- **Active**: true/false - activates or deactivates the model
- **Configs**: List of config names (only for to_train and to_cross_validate)
- **Names**: List of model names

## algorithm_base_path

Here you specify the paths from where models should be loaded if needed.
Each entry points to the folder where the model to be loaded is located. The file name is stored under "models" in the "Names" list of the respective phase. The entry must bear the class name of the algorithm to which the model belongs.
If the path is set to "default", it will be set to the standard output directory specified under "storage_path".

## storage_path
This path specifies where the standard output directory is located, in which models, data, metadata, and results are saved.
If this entry is not set, the standard output directory will be set to the folder "Save/%Y-%m-%d_%H-%M-%S" in the current working directory.

## data

This defines where the patient data is located that should be loaded and used for training or classification/evaluation.

### Time Series Data
- **timeseries_file_path**: File path for time series models. Various formats are currently supported (see import_type).
- **import_type**: Defines the format of input data. Options:
  - **numpy**: .npy files (also requires .vars file in the same folder)
  - **csv**: CSV files in the format of the Data-Extractor
  - **pkl**: Pickle files
  - **split**: Pre-split datasets
- **database**: Specifies which database the data comes from. This is relevant for unit conversion (see perform_unit_conversion). Supported values: "eICU", "MIMIC3", "MIMIC4", "UKA", "CALIBRATION", "CONTROL"

### Image Data
- **image_file_path**: Path to the folder with image data datasets. This folder must have the following structure:
  - A "chexpert" folder and a "mimic" folder, each containing the training and (for mimic) the ARDS test dataset
  - Each dataset consists of an "image" folder (images) and a "label" folder (labels)
  - A "models" folder for trained models with subfolders for each model containing "pneumonia" and "ards" folders, each with a "main" folder
  - A "results" folder with the same structure as "models" for evaluation results
  - A file "aug_tech.txt" with augmentation techniques (one per line)

- **pneumonia_image_dataset**: Name of the dataset for pneumonia training (e.g., "balanced")
- **ards_image_dataset**: Name of the dataset for ARDS training (e.g., "weighted")

Note: The image data structure is complex due to various datasets with different weightings and balancing from previous work.

## preprocessing

In this section, some settings are made that precisely define how data preprocessing proceeds.

* **patients_per_process**: Specifies how many patients should be processed in one process
* **max_processes**: Specifies how many processes should run in parallel at most

### anomaly_detection
Here you configure the anomaly detection algorithm to be used. Each algorithm has its own section. Currently supported algorithms:
* Physiological Limits (Physiological_Outliers)
* SW-ABSAD-MOD
* DeepAnt
* ALAD

Each of these algorithms has the following configuration options:
* **active**: Specifies whether the algorithm should be used for anomaly detection. The first algorithm with active set to true will always be used
* **name**: Name of the algorithm as it should appear in metadata
* **columns_to_check**: List of which columns should be checked for anomalies. Depending on the AD algorithm used, different inputs are expected here. For details, please check the implementation. If set to [], all columns except patient_id and timestamp will be checked.
* **handling_strategy**: How should detected anomalies be handled. Possible values:
  * **delete_value**: Deletes the value detected as an anomaly
  * **delete_than_impute**: Deletes the value and then imputes it as described under **fix_algorithm**
  * **delete_row_if_any_anomaly**: Deletes the entire row if any value is marked as an anomaly
  * **delete_row_if_many_anomalies**: Deletes the entire row if the number of values marked as anomalies exceeds **anomaly_threshold**. If the row is not deleted, values detected as anomalies will be imputed with the algorithm specified under **fix_algorithm**
* **fix_algorithm**: Specifies how values deleted due to AD should be imputed. Possible values:
  * **forward**: Last non-deleted value is filled forward
  * **backward**: Next non-deleted value is filled backward
  * **interpolate**: Linear interpolation between last and next value. Warning: Value is influenced by the number of missing data points
* **anomaly_threshold**: Relative number of anomalies above which a row is deleted when **delete_row_if_many_anomalies** is selected
* **supported_stages**: Specifies which work steps are supported by the AD algorithm. This entry should only be changed if additional work steps are implemented or removed for a model. Possible values:
  * **prepare**: Work step that prepares the data for processing by the algorithm
  * **train**: Work step that trains the deep learning models
  * **predict**: Work step that detects anomalies based on the chosen algorithm
  * **fix**: Work step that handles detected anomalies with the algorithm specified under **handling_strategy**
* **active_stages**: Work steps to be executed when the chosen algorithm is run. This is a list. Possible entries are described under **supported_stages**.
* **anomaly_data_dir**: Folder where the results of the **predict** step are saved
* **prepared_data_dir**: Folder where the results of the **prepare** step are saved

After that follow algorithm-specific settings. For an explanation of these, check either the implementation or the original papers.
Deep learning approaches additionally have the following settings:

* **run_dir**: Training and AD step logs are saved here
* **checkpoint_dir**: Models for the checkpoints specified in the code are saved here
* **load_data**: Specifies whether existing data should be used in the prepare step or not
* **save_data**: Specifies whether the training/test/eval datasets generated in the prepare step should overwrite already existing datasets
* **retrain_models**: Dictionary that specifies whether already existing models should be used or new models should be trained. For possible values, please check the implementation.

### filtering
Here you define which filters for filtering patients who may have an incorrect ARDS label should be activated. Possible filters are Strict, Lite, and BD (each as its own bullet point under "filter").

* **Strict**: Filter Strict removes all patients from the data who supposedly don't have ARDS but for whom a Horowitz quotient below 200 mmHg was recorded.
* **Lite**: Filter Lite removes all patients from the data who supposedly don't have ARDS and additionally have neither hypervolemia, nor pulmonary edema, nor heart failure, but still have a Horowitz quotient below 200 mmHg. Filters Strict and Lite should sensibly not be used simultaneously, as Filter Lite is similar to Strict, just slightly less restrictive.
* **BD**: Filter BD (Berlin Definition) removes all patients from the data who supposedly have ARDS but for whom a Horowitz quotient below 300 mmHg was never measured (contradicts the ARDS definition).

### imputation
Here you define for which parameters missing data should be imputed and which imputation algorithm should be used for each parameter. The following algorithms are available:
- **forward**: Missing values are filled with the last known value. If missing values appear at the very beginning, where there is no previous known value, the first available value is used for these gaps.
- **backfill**: Missing values are filled with the next known value. If missing values appear at the very end, where there is no next known value, the last available value is used for these gaps.
- **mean**: The average of all available values is calculated. All gaps are filled with this average.
- **linear_interpolation**: Missing values are calculated by linear interpolation between the two nearest neighbors. Edge values are filled by forward or backfill.

Some parameters represent a binary state, e.g., because they indicate whether a certain diagnosis was made or not. Currently, these are the parameters "ards", "heart-failure", "hypervolemia", "mech-vent", "pneumonia", "xray", "sepsis", "chest-injury". For these, only the methods "forward" or "backfill" may be used.

* **impute_empty_cells**: Defines what should happen to entries that still have empty values after imputation. This can happen when certain values are completely missing for a patient. If true, all missing values in such columns are set to -100000. If false, all columns containing only NaN values and all rows containing at least one NaN value are deleted.
* **default_imputation_method**: Specifies the default imputation method used for all parameters to be imputed for which no separate method was specified.
* **params_to_impute**: Lists all parameters to be imputed. If "all" is specified here, all parameters will be imputed. For each parameter, the imputation algorithm to be used specifically for that parameter can also be specified. The syntax for this looks like: "- ards, forward".

Missing values in the data can lead to program crashes. Therefore, it makes sense to always impute all parameters.

### params_to_calculate
Here you specify which parameters should be calculated from the available data. Currently, the following parameters are possible:
- delta-p (requires parameters "p-ei" and "peep" in the data)
- tidal-vol-per-kg (requires parameters "height", "weight", and "tidal-volume". The parameter "gender" would also be desirable. If gender is not specified, the formula for male gender is used)
- liquid-balance (requires parameters "liquid-input" and "liquid-output")
- lymphocytes_abs (requires parameters "lymphocytes (percentage)" and "leucocytes")
- horovitz (requires parameters "fio2" and "pao2")
- i-e (requires parameters inspiry-time and expiry-time)
- lymphocytes (relative) (requires parameters lymphocytes_abs and leucocytes)

Parameters to be calculated that are already present in the data will be skipped.
If one of the parameters needed to calculate a new parameter is not given in the data, the framework will output an error message.

### ards_onset_detection
Here you set according to which rule the onset of an ARDS course should be detected and which data exactly should be returned.
- **detection_rule**: This is the rule by which ARDS onset is detected. The following options are available:
  - *lowest_horovitz*: The time point with the lowest Horowitz quotient of a patient
  - *first_horovitz*: The first Horowitz quotient of a patient that is below 300 mmHg. If there is none, the lowest Horowitz quotient is taken.
  - *4h*, *12h*, *24h*: The first time point at which the patient's Horowitz quotient is below 300 mmHg for the next 4/12/24 hours. If this condition cannot be met, the beginning of the time point from which the Horowitz quotient is below 300 for the longest time is taken.
  - *4h_50*: The first time point at which at least 50% of the patient's Horowitz values for the next 4 hours are below 300 mmHg. If there is no such series, the beginning of the series is taken where the percentage of Horowitz quotients below 300 for the next 4 hours is highest.
- **return_rule**: This specifies which data exactly should be returned. The following rules are available:
  - *datapoint*: For each patient, only the exact time point of the determined ARDS onset is returned, all other rows are not further processed.
  - *data_series_as_series*: All rows for the patient that lie between two defined points are returned. The two boundary points are defined via "series_start_point" and "series_end_point", see below.
  - *data_series_as_point*: As with *data_series_as_series*, first all rows that lie between the two defined points are identified. Then all these rows are merged into a single dataset. That means, from the following series:
  
    | patient_id | time | horovitz | peep |
    |---|---|---|---|
    | 1 | 0 | 300 | 5 |
    | 1 | 5 | 250 | 6 |
    
    the following row would result:
    
    | patient_id | horovitz1 | peep1 | horovitz2 | peep2 |
    |---|---|---|---|---|
    | 1 | 300 | 5 | 250 | 6 |
    
    The columns "time", "patient_id" and "ards" are not duplicated. Under "time", only the exact time point of the detected ARDS onset is subsequently stored.
- **series_start_point** and **series_end_point**: These define from where to where the returned series should be, if *data_series_as_series* or *data_series_as_point* was chosen as return rule. Both specify the beginning/end of the series relative to the detected ARDS onset in seconds. I.e., if series_start_point is set to -200000, all measured values up to 200000 seconds before the detected ARDS onset are considered.
  If "datapoint" was chosen as return rule, it doesn't matter what is entered here.
- **remove_ards_patients_without_onset**: It can happen that patients in the data were marked as ARDS patients, but no ARDS onset is found according to the above rule. This parameter controls what should happen to these patients - if true, these patients are removed from the data. Non-ARDS patients are not affected by this.
- **impute_missing_rows**: If "data_series_as_series" or "data_series_as_point" is chosen as return rule, it can happen that the chosen return period is too large for the available data. If this point is set to true, any missing rows will be added and the corresponding data imputed with the values -100000.
- **update_ards_values**: If true and if *data_series_as_series* was chosen as return rule, each patient's ARDS value before the detected ARDS onset is set to 0 and from the detected ARDS onset onwards to 1.

The algorithm will also attempt to find an ARDS onset for non-ARDS patients according to the chosen rule. Often, of course, nothing should be found. In that case, the above rules are followed to determine a time point. Data is also returned for non-ARDS patients according to the rule defined under "return_rule".

## feature_selection

Here you can choose the method to be used for feature selection. Depending on the method, some additional parameters are needed.
- **method**: Here you specify the exact method. Available options:
  - *low_variance*: Parameters whose values have low variance are removed. The variance threshold is specified under "variance".
  - *univariate*: Selects the k best parameters according to a univariate statistical test. The value of k must be defined here, see below.
  - *recursive*: Recursive feature elimination. The desired number of features to be retained can be defined via "k". If k is not defined, a number recommended by the feature selection method will be returned.
  - *recursive_with_cv*: Also recursive feature elimination, but additionally uses cross-validation to find the optimal number of features. Via "k", the desired number of features can also be specified here (but doesn't have to be).
  - *L1*: L1-based feature selection
  - *tree*: Tree-based feature selection
  - *sequential*: Sequential feature selection
- **variance**: Here you specify the variance below which parameters should be removed when the "low_variance" method is used.
- **k**: Here you specify the target number of parameters you want to use. Considered by the methods "univariate", "recursive", and "recursive_with_cv". In the former case, the specification is mandatory. In the latter two cases, the specification can be omitted, then the method itself chooses the optimal number of parameters.

## data_segregation

Here you define exactly how the loaded data should be split into training and test data.
- **training_test_ratio**: Here you specify what proportion of the data should be used for training. So if 0.8 is specified, 80% of patients will be used for training and 20% for testing.
- **percentage_of_ards_patients**: This specifies what the proportion of ARDS patients should be. This applies to both the training dataset and the test dataset. This setting will likely result in data having to be removed to achieve the required ratio.
- **splitting_seed**: Specifies a seed used for splitting the data. Allows reproducibility.

## evaluation

- **cross_validation**: Here you can make some settings for cross-validation.
  - **n_splits**: Here you specify the number of subsets into which the test data should be divided for cross-validation.
  - **random_state**: Seed for reproducibility
  - **shuffle**: (true or false) - specifies whether the data should be mixed before splitting.
- **threshold_optimization_algorithms**: List of algorithms that each determine an optimal decision boundary. Options include:
  - **Standard**: Uses default threshold (typically 0.5)
  - **MaxTPR**: Maximizes True Positive Rate
  - **MaxTPRMinFPR**: Maximizes TPR while minimizing False Positive Rate
  - **GeometricRoot**: Uses geometric mean of sensitivity and specificity
- **evaluation_metrics**: List of names of metrics to be calculated during evaluation. Available metrics include:
  - **AUC**: Area Under the ROC Curve
  - **Accuracy**: Overall accuracy
  - **F1Score**: F1 Score
  - **FPR**: False Positive Rate
  - **MCC**: Matthews Correlation Coefficient
  - **OptimalProbability**: Optimal probability threshold
  - **Sensitivity**: True Positive Rate / Recall
  - **Specificity**: True Negative Rate
  - **TPR**: True Positive Rate
  - **PPV**: Positive Predictive Value / Precision
  - **NPV**: Negative Predictive Value
  - **Precisions**: Precision values at different thresholds
  - **Recalls**: Recall values at different thresholds
- **evaluation_name**: Name under which the evaluation result should be displayed and saved

## image_model_parameters

Here you can define parameters that control the training process for image data models.

### General Parameters
- **method**: Defines which layers of the pretrained model are "unfrozen" for fine-tuning. Options:
  - **last_layer**: Only the last layer is unfrozen
  - **last_block**: The last block is unfrozen
  - **model**: The complete model is unfrozen

- **mode**: Controls the use of data augmentation. Options:
  - **mode1**: No augmentation - training with original images
  - **mode2**: Reserved for future use (augmentation only during training via transforms)
  - **mode3**: ARDS training data uses pre-augmented datasets (MIMIC-DB-AUG folder)
  - **mode4**: ARDS training AND test data use pre-augmented datasets
  
  Note: PNEUMONIA training always uses non-augmented data (CheXpert-DB-PNEUMONIA)

### CNN Models (ResNet, DenseNet)

**Pretraining Parameters:**
- **learning_rate_pre_cnn**: Learning rate for pretraining (e.g., 0.001)
- **batch_size_pre_cnn**: Batch size for pretraining (e.g., 64)
- **weight_decay_pre_cnn**: Weight decay for regularization (e.g., 0.00001)

**Main Training Parameters:**
- **learning_rate_cnn**: Learning rate for main training (e.g., 0.05)
- **epoch_decay_cnn**: Learning rate decay per epoch (e.g., 0.002)
- **weight_decay_cnn**: Weight decay for regularization (e.g., 0.00001)
- **margin_cnn**: Margin parameter for loss function (e.g., 1.0)

**Pneumonia Pretraining:**
- **num_epochs_pneumonia**: Number of training epochs for pneumonia model
- **batch_size_pneumonia**: Batch size for pneumonia training (e.g., 32)
- **SEED_pneumonia**: Seed for reproducibility (e.g., 123)

**ARDS Fine-Tuning:**
- **num_epochs_ards**: Number of training epochs for ARDS model
- **batch_size_ards**: Batch size for ARDS training (e.g., 32)
- **SEED_ards**: Seed for reproducibility (e.g., 105)

**Cross-Validation:**
- **k_folds**: Number of folds for cross-validation
  - 1 = simple 80/20 split (faster)
  - 2+ = k-fold cross-validation

### ViT Models (Vision Transformer)
- **learning_rate_vit**: Learning rate for ViT models (e.g., 0.0005)
- **batch_size_vit**: Batch size for ViT training (e.g., 32)
- **k_folds_vit**: Number of folds for cross-validation with ViT
