import yaml


def config_generator(template_config_path: str, filtering: bool, splitting_seed: int, ad_active: bool, detector: str=None, imputation: str=None):
    with open(template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    name = f"{'None' if not filtering else 'Strict'} {detector if ad_active else 'No AD'} {imputation if ad_active else ''} {splitting_seed} Lowest Horovitz"
    config_name = f"config_{name.replace(' ', '_')}.yml"
    template_config["process"]["perform_filtering"] = filtering
    template_config["process"]["perform_anomaly_detection"] = ad_active
    if ad_active:
        for key, value in template_config["preprocessing"]["anomaly_detection"].items():
            if key == detector:
                template_config["preprocessing"]["anomaly_detection"][key]["active"] = True
                template_config["preprocessing"]["anomaly_detection"][key]["fix_algorithm"] = imputation
            else:
                template_config["preprocessing"]["anomaly_detection"][key]["active"] = False
    template_config["evaluation"]["cross_validation"]["random_state"] = splitting_seed
    template_config["evaluation"]["evaluation_name"] = f"Evaluation {name}"
    yaml.dump(template_config, open(config_name, 'w'))
    return template_config

def config_generator_without_fix(template_config_path: str, filter_active: bool, splitting_seed: int, detector: str=None, imputation: str=None):
    with open(template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    name = f"{'None' if not filter_active  else 'Strict'} No AD {splitting_seed} Lowest Horovitz"
    config_name = f"config_{name.replace(' ', '_')}.yml"
    template_config["process"]["perform_filtering"] = filtering
    template_config["process"]["perform_anomaly_detection"] = False
    template_config["evaluation"]["cross_validation"]["random_state"] = splitting_seed
    template_config["evaluation"]["evaluation_name"] = f"Evaluation {name}"
    if not detector:
        data_path = "/work/rwth1474/Data/timeseries_data/uka_data_050623_testing_patients.npy"
    else:
        data_path = f"/work/rwth1474/Data/AnomalyDection/anomaly_data/{detector}/fixed_data_{detector}_delete_than_impute_{imputation}.pkl"
    template_config["data"]["timeseries_file_path"] = data_path
    yaml.dump(template_config, open(config_name, 'w'))
    return template_config

if __name__ == "__main__":
    filtering = [True, False]
    detectors = ["SW_ABSAD_MOD", "DeepAnt", "Physical_Outliers", "No_AD"]
    imputations = ["forward", "interpolate"]
    fix_data = False
    seed = 69
    for filter_active in filtering:
        for detector in detectors:
            for imputation in imputations:
                if detector == "No_AD":
                    if fix_data:
                        config_generator("config.yml", filter_active, seed, False)
                    else:
                        config_generator_without_fix("config.yml", filter_active, seed)
                else:
                    if fix_data:
                        config_generator("config.yml", filter_active, seed, True, detector, imputation)
                    else:
                        config_generator_without_fix("config.yml", filter_active, seed, detector, imputation)


