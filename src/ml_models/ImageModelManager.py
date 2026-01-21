import importlib
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ImageModelManager:
    """
    Manager for image model creation and loading.
    Mirrors the responsibilities of TimeSeriesModelManager but adapted for image models:
      - create models from config (instantiation + hyperparameter loading)
      - set storage locations
      - load models from disk when required

    Expected `needed_models` structure (same style as timeseries manager):
    {
        "ResNet": {
            "Names": ["resnet-small", ...],
            "Configs": ["default", "custom_resnet.yml", ...]
        },
        "VisionTransformer": { ... }
    }

    The manager will try several common lookup patterns to instantiate classes:
      - ml_models.<module>.<ClassName> where module = model_type.lower()
      - Class name patterns: <ModelType>ImageModel, <ModelType>Model
      - fallback: eval("<ModelType>ImageModel()") as last resort (keeps parity with timeseries manager)
    """

    def __init__(self, config: dict, outdir: str | Path):
        logger.info("Initializing ImageModelManager...")
        self.config = config
        self.save_models = bool(config.get("process", {}).get("save_models", False))
        self.outdir = Path(outdir).resolve()
        # ensure base output directory exists
        self.outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save models enabled: {self.save_models}")
        logger.info(f"Image models output directory: {self.outdir}")

    def _instantiate_class_by_name(self, model_type: str):
        """Try to import/instantiate an image model class for model_type."""
        candidates = [
            (model_type.lower(), model_type + "ImageModel"),
            (model_type.lower(), model_type + "Model"),
            (model_type.lower(), model_type),  # class may be same as module name
        ]
        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(f"ml_models.{module_name}")
                cls = getattr(module, class_name, None)
                if cls:
                    logger.debug(f"Found image model class '{class_name}' in ml_models.{module_name}")
                    return cls
            except ModuleNotFoundError:
                continue
            except Exception as e:
                logger.debug(f"Error importing ml_models.{module_name}: {e}")
                continue

        # last resort: try to eval "<model_type>ImageModel()"
        try:
            instance = eval(model_type + "ImageModel()")
            logger.debug("Instantiated image model via eval fallback")
            return instance.__class__
        except Exception as e:
            logger.error(f"Could not find or instantiate image model class for '{model_type}': {e}")
            return None

    def create_models_from_config(self, needed_models: dict, base_config_path: str):
        """
        Create image model instances according to needed_models specification.
        Loads hyperparameters from YAML config files if provided (structure parallels TimeSeries manager).
        Returns dict mapping model_type -> list(instances).
        """
        logger.info("=" * 80)
        logger.info("Creating Image Models from Configuration")
        logger.info("=" * 80)

        models = {}
        total_models = sum(len(needed_models[model_type]["Names"]) for model_type in needed_models)
        current = 0

        for model_type, spec in needed_models.items():
            names = spec.get("Names", [])
            configs = spec.get("Configs", ["default"] * len(names))
            logger.info(f"Creating {len(names)} models of type: {model_type}")

            cls = self._instantiate_class_by_name(model_type)
            if cls is None:
                logger.error(f"No class available for image model type '{model_type}', skipping these entries")
                continue

            for i, model_name in enumerate(names):
                current += 1
                logger.info(f"[{current}/{total_models}] Creating image model '{model_name}' ({model_type})...")
                try:
                    model = cls() if callable(cls) else cls
                except Exception as e:
                    logger.error(f"Failed to instantiate {model_type} class: {e}")
                    raise

                # assign name & algorithm if available
                try:
                    model.name = model_name
                except Exception:
                    setattr(model, "name", model_name)

                try:
                    model.algorithm = model_type
                except Exception:
                    setattr(model, "algorithm", model_type)

                # load hyperparameters if provided
                # Currently not implemented/tested for image models, but structure is here
                cfg = configs[i] if i < len(configs) else "default"
                if cfg != "default":
                    hyperparameters_path = Path(base_config_path) / str(model_type) / cfg
                    logger.info(f"Loading image model hyperparameters from: {hyperparameters_path}")
                    if not hyperparameters_path.exists():
                        logger.warning(f"Hyperparameter file not found: {hyperparameters_path} — using defaults")
                    else:
                        try:
                            with open(hyperparameters_path, "r") as f:
                                hparams = yaml.safe_load(f) or {}
                            if hasattr(model, "set_params"):
                                model.set_params(hparams)
                            else:
                                # attach params if setter missing
                                setattr(model, "params", hparams)
                            logger.info(f"Loaded hyperparameters for {model_name}: {list(hparams.keys())}")
                        except Exception as e:
                            logger.warning(f"Failed to load hyperparameters {hyperparameters_path}: {e}")

                # set storage location
                if self.save_models:
                    storage = (self.outdir / f"{model.algorithm}_{model.name}").resolve()
                    # create storage directory
                    storage.mkdir(parents=True, exist_ok=True)
                    model.storage_location = str(storage)
                    logger.debug(f"Model storage location set to: {model.storage_location}")
                else:
                    model.storage_location = "Model is not saved"
                    logger.debug("Model persistence disabled (save_models=False)")

                models.setdefault(model_type, []).append(model)
                logger.info(f"[{current}/{total_models}] Created image model '{model_name}'")

        logger.info("=" * 80)
        logger.info("Image model creation completed")
        logger.info(f"Total image models created: {sum(len(v) for v in models.values())}")
        logger.info("=" * 80)
        return models

    def load_models(self, needed_models: dict, available_models_dict: dict, model_base_paths: dict):
        """
        Load image models from disk if not already available in memory.

        Args:
            needed_models: same structure as create_models_from_config
            available_models_dict: dict with lists of already available models per type
            model_base_paths: mapping model_type -> base path (or 'default' to use self.outdir)

        Returns:
            updated available_models_dict with loaded models appended
        """
        logger.info("=" * 80)
        logger.info("Loading Image Models from Configuration")
        logger.info("=" * 80)

        total_needed = sum(len(needed_models[mt]["Names"]) for mt in needed_models)
        loaded = 0
        already = 0

        for model_type, spec in needed_models.items():
            names = spec.get("Names", [])
            available_list = available_models_dict.setdefault(model_type, [])

            base_path = Path(model_base_paths.get(model_type, "default"))
            if str(base_path) == "default":
                base_path = self.outdir

            logger.debug(f"Base load path for {model_type}: {base_path}")

            for model_name in names:
                # check in-memory
                in_memory = any(getattr(m, "name", None) == model_name for m in available_list)
                if in_memory:
                    already += 1
                    logger.info(f"Image model '{model_name}' ({model_type}) already in memory")
                    continue

                # try to load from disk: instantiate class then call its load method
                cls = self._instantiate_class_by_name(model_type)
                if cls is None:
                    logger.error(f"Cannot load model '{model_name}': no class for type {model_type}")
                    continue

                try:
                    instance = cls() if callable(cls) else cls
                    instance.name = model_name
                    instance.algorithm = model_type
                    # construct storage path and ensure it exists
                    candidate_path = (Path(base_path) / f"{model_type}_{model_name}").resolve()
                    if not candidate_path.exists():
                        logger.warning(f"Model path not found for '{model_name}': {candidate_path}")
                        raise FileNotFoundError(str(candidate_path))
                    # expect model to implement a 'load' or 'load_model' method
                    if hasattr(instance, "load"):
                        instance.load(str(candidate_path))
                    elif hasattr(instance, "load_model"):
                        instance.load_model(str(candidate_path))
                    else:
                        logger.error(f"Image model class for '{model_type}' has no load() method")
                        raise RuntimeError("No load method")
                    available_list.append(instance)
                    loaded += 1
                    logger.info(f"Loaded image model '{model_name}' ({model_type}) from {candidate_path}")
                except Exception as e:
                    logger.error(f"Failed to load image model '{model_name}' ({model_type}): {e}")

        logger.info("=" * 80)
        logger.info("Image model loading finished")
        logger.info(f"Total image models needed: {total_needed}")
        logger.info(f"Image models loaded from disk: {loaded}")
        logger.info(f"Image models already in memory: {already}")
        logger.info("=" * 80)
        return available_models_dict