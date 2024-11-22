import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Hyperparameters')

	parser.add_argument('--emb_size', default=64, type=int, help='embedding size')
	parser.add_argument('--n_layers', default=3, type=int, help='gnn layers')
	parser.add_argument('--topks', default=10, type=int, help='@k test list')
	parser.add_argument('--data_dir', default='./data/', type=str, help='dataset directory')
	parser.add_argument('--target', default='buy', type=str, help='target behavior')
	return parser.parse_args()
args = parse_args()
