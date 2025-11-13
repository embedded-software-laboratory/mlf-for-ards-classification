from argparse import ArgumentParser


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog='ml-framework',
        description='A framework for the development of AI models used for ' +
        'ARDS classification',
    )

    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '-f',
        '--config-file',
        type=str,
        default='config.yml',
        help='Path to the configuration file',
    )
    config_group.add_argument(
        '-c',
        '--config',
        type=str,
        help='JSON string of the configuration',
    )

    return parser
