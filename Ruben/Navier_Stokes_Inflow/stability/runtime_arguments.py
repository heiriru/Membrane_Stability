import argparse

parser = argparse.ArgumentParser()
parser.add_argument("problem")
parser.add_argument("input_directory")
parser.add_argument("output_directory")
# unknown (script-specific) options, e.g. --Re, are parsed by the individual scripts
args, unknown_args = parser.parse_known_args()
