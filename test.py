import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model import TextVAE
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--hidden_size', type=int, default=512,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="number of hidden units for RNN")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--dropword', type=float, default=0,
                    help="dropout applied to input tokens (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--kla', action='store_true',
                    help="use kl annealing")
parser.add_argument('--bow', action='store_true',
                    help="add bag of words loss in training")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)


def loss_function(targets, outputs, mu, logvar, bow=None):
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # bow: from batch_size x vocab_size to batch_size * sequence_length x vocab_size
    if bow is None:
        bow_loss = torch.tensor(0., device=outputs.device)
    else:
        bow = bow.unsqueeze(1).repeat(1, outputs.size(1), 1).contiguous()
        bow_loss = F.cross_entropy(bow.view(bow.size(0) * bow.size(1),
                                            bow.size(2)),
                                   targets.view(-1),
                                   size_average=False,
                                   ignore_index=PAD_ID)
    return ce_loss, kld, bow_loss


def get_loss(texts, lengths, model, device):
    inputs = texts[:, :-1].clone().to(device)
    targets = texts[:, 1:].clone().to(device)
    outputs, mu, logvar, bow = model(inputs, lengths)
    if not args.bow:
        bow = None
    return loss_function(targets, outputs, mu, logvar, bow)


def evaluate(data_source, model, device):
    model.eval()
    total_ce = 0.0
    total_kld = 0.0
    total_bow = 0.0
    total_words = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, lengths, _ = data_source.get_batch(batch_size, i)
        ce, kld, bow_loss = get_loss(texts, lengths, model, device)
        total_ce += ce.item()
        total_kld += kld.item()
        total_bow += bow_loss.item()
        total_words += sum(lengths)
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_bow / data_source.size, ppl)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    path = './saves/emb{0:d}.hid{1:d}{2}{3}{4}.{5}.pt'.format(
        args.embed_size, args.hidden_size,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        '.kla' if args.kla else '', '.bow' if args.bow else '', dataset)
    return path


def main(args):
    print("Loading data")
    corpus = data.Corpus(args.data, max_vocab_size=args.max_vocab,
                         max_length=args.max_length)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train_data.size)
    print("\tvocabulary size: ", vocab_size)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)
    test_ce, test_kld, test_bow, test_ppl = evaluate(corpus.test_data, model, device)
    print('=' * 90)
    print("| End of training | test loss {:5.2f} ({:5.2f}) "
          "| test ppl {:5.2f} | bow loss {:5.2f}".format(
              test_ce, test_kld, test_ppl, test_bow))
    print('=' * 90)


if __name__ == '__main__':
    main(args)
