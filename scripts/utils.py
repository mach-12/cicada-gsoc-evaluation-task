import yaml


def save_params_to_yaml(params, filename="best_model_params.yaml"):
    """Save dictionary parameters to a YAML file."""
    with open(filename, "w") as file:
        yaml.dump(params, file)


def load_params_from_yaml(filename="best_model_params.yaml"):
    """Load dictionary parameters from a YAML file."""
    with open(filename, "r") as file:
        return yaml.safe_load(file)
