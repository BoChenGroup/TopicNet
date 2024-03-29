import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from mydataset import *
from model import *
import pickle
from trainer import GBN_trainer
import argparse
import scipy.io as sio # mat

topic_list = []
graph_wordnet = sio.loadmat('./dataset/TopicTree_20ng.mat')

for i in range(len(graph_wordnet['graph_topic_name'][0])):
    topic_list.append(len(graph_wordnet['graph_topic_name'][0][i]))

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=200, help="models used.")
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=1, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=topic_list[1:8], help='Number of units in hidden layer 1.') # [512, 256, 128, 64, 32, 16, 8]
parser.add_argument('--hidden_size', type=int, default=500, help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=int, default=50, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-dir', type=str, default='./dataset/20ng.pkl', help='type of dataset.')
parser.add_argument('--output-dir', type=str, default='torch_phi_output_etm_hier_share', help='type of dataset.')
parser.add_argument('--save-path', type=str, default='saves/gbn_model_weibull_etm_share_50_7_kl_0.1', help='type of dataset.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


train_loader, vocab_size, voc = get_loader_txt(args.dataset_dir, batch_size=args.batch_size)
test_loader, vocab_size, voc = get_test_loader_txt(args.dataset_dir, batch_size=args.batch_size)

args.vocab_size = vocab_size

args.n_updates = len(train_loader) * args.epochs
args.MBratio = len(train_loader)

trainer = GBN_trainer(args,  voc_path=voc)
trainer.train(train_loader, test_loader)
trainer.vis_txt()
