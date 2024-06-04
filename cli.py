import argparse

def make_parser():
    parser = argparse.ArgumentParser(description="MLP Framework")

    # Exclusive group for config
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", "-c", help="config as json string")
    group.add_argument("--config_file", "-f", help="config file path")

    return parser
