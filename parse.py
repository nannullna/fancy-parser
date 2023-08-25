from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config file.')
args, remaining_args = parser.parse_known_args(["--epochs", "20"])

print(args)
print(remaining_args)

