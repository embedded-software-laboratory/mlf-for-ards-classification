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
    template_config["data_segregation"]["splitting_seed"] = splitting_seed
    template_config["evaluation"]["evaluation_name"] = f"Evaluation {name}"
    yaml.dump(template_config, open(config_name, 'w'))
    return template_config

if __name__ == "__main__":
    filtering = [True, False]
    detectors = ["SW_ABSAD_MOD", "DeepAnt", "Physical_Outliers", "No_AD"]
    imputations = ["forward", "interpolate"]
    for filter_active in filtering:
        for detector in detectors:
            for imputation in imputations:
                if detector == "No_AD":
                    config_generator("config.yml", filter_active, 69, False)
                else:
                    config_generator("config.yml", filter_active, 69, True, detector, imputation)


