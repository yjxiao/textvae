import argparse
import time
import math
import torch
import torch.nn.functional as F

from model import LM
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text LM')
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
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)


def loss_function(targets, outputs):
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    return ce_loss


def get_loss(texts, lengths, model, device):
    inputs = texts[:, :-1].clone().to(device)
    targets = texts[:, 1:].clone().to(device)
    outputs = model(inputs, lengths)
    return loss_function(targets, outputs)


def evaluate(data_source, model, device):
    model.eval()
    total_ce = 0.0
    total_words = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, lengths, _ = data_source.get_batch(batch_size, i)
        ce_loss = get_loss(texts, lengths, model, device)
        total_ce += ce_loss.item()
        total_words += sum(lengths)
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, ppl)


def train(data_source, model, optimizer, epoch, device):
    model.train()
    total_ce = 0.0
    total_words = 0
    for i in range(args.epoch_size):
        texts, lengths, _ = data_source.get_batch(args.batch_size)
        ce_loss = get_loss(texts, lengths, model, device)
        total_ce += ce_loss.item()
        total_words += sum(lengths)
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, ppl)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    path = './saves/emb{0:d}.hid{1:d}{2}.{3}.lm.pt'.format(
        args.embed_size, args.hidden_size,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        dataset)
    return path


def main(args):
    print("Loading data")
    corpus = data.Corpus(args.data, max_vocab_size=args.max_vocab,
                         max_length=args.max_length)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train_data.size)
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    model = LM(vocab_size, args.embed_size, args.hidden_size,
               args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_ce, train_ppl = train(corpus.train_data, model, optimizer, epoch, device)
            valid_ce, valid_ppl = evaluate(corpus.valid_data, model, device)
            print('-' * 70)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} | train ppl {:5.2f}".format(
                      train_ce, train_ppl))
            print(len(meta) * ' ' + "| valid loss {:5.2f} "
                  "| valid ppl {:5.2f}".format(
                      valid_ce, valid_ppl), flush=True)
            if best_loss is None or valid_ce < best_loss:
                best_loss = valid_ce
                with open(get_savepath(args), 'wb') as f:
                    torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 70)
        print('Exiting from training early')


    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)
    test_ce, test_ppl = evaluate(corpus.test_data, model, device)
    print('=' * 70)
    print("| End of training | test loss {:5.2f} | test ppl {:5.2f}".format(
              test_ce, test_ppl))
    print('=' * 70)


if __name__ == '__main__':
    main(args)
