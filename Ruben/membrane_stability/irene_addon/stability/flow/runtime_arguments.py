import argparse

parser = argparse.ArgumentParser()
parser.add_argument("problem")
parser.add_argument("input_directory")
parser.add_argument("output_directory")
# script-specific options (e.g. --h) are parsed by the individual scripts
args, unknown_args = parser.parse_known_args()
