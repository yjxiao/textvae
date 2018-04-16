import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='Text VAE translate')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the corpus (same as training)')
parser.add_argument('--checkpoint', type=str, default='./saves/model.pt',
                    help='location of the model file')
parser.add_argument('--input_file', type=str, default='./data/test.txt',
                    help='location of the input texts')
parser.add_argument('--output_file', type=str, default='./data/test.txt.out',
                    help='output file to write reconstructed texts')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size used in generation')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def predict(data_source, model):
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, _ = data_source.get_batch(i, batch_size)
        if args.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs, mu, logvar = model(inputs)
        


with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

print("Loading data")
corpus = data.Corpus(args.data)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
# data to be reconstructed
input_data = data.Data(args.input_file, corpus.word2idx)
