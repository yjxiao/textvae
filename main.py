import argparse
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model import TextVAE
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data',
                    help="location of the data folder")
parser.add_argument('--embed_size', type=int, default=256,
                    help="size of the word embedding")
parser.add_argument('--hidden_size', type=int, default=128,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="number of hidden units for RNN")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--dropword', type=float, default=0.2,
                    help="dropout applied to input tokens (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-2,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=1e-4,
                    help="weight decay used for regularization")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
parser.add_argument('--save', type=str,  default='./saves/model.pt',
                    help="path to save the final model")
args = parser.parse_args()

torch.manual_seed(args.seed)


def loss_function(targets, outputs, mu, logvar):
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ce_loss, kld


def evaluate(data_source, model):
    model.eval()
    total_ce = 0.0
    total_kld = 0.0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, targets, lengths = data_source.get_batch(i, batch_size)
        if not args.nocuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs, mu, logvar = model(inputs, lengths)
        ce, kld = loss_function(targets, outputs, mu, logvar)
        total_ce += ce.data[0]
        total_kld += kld.data[0]

    return total_ce / data_source.size, total_kld / data_source.size


def train(data_source, model, optimizer, epoch):
    model.train()
    total_ce = 0.0
    total_kld = 0.0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, targets, lengths = data_source.get_batch(i, batch_size)
        if not args.nocuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs, mu, logvar = model(inputs, lengths)
        ce, kld = loss_function(targets, outputs, mu, logvar)
        total_ce += ce.data[0]
        total_kld += kld.data[0]
        kld_weight = weight_schedule(data_source.size, args.batch_size, epoch, i)
        optimizer.zero_grad()
        loss = ce + kld_weight * kld
        loss.backward()
        optimizer.step()
    return total_ce / data_source.size, total_kld / data_source.size, kld_weight


def interpolate(i, n):
    return min(i / n, 1)


def weight_schedule(data_size, batch_size, epoch, i):
    """Scheduling of the KLD annealing weight. """
    return interpolate((data_size // batch_size) * (epoch - 1) + i // batch_size, 50000)


print("Loading data")
corpus = data.Corpus(args.data)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
print("Constructing model")
print(args)
model = TextVAE(vocab_size, args.embed_size, args.hidden_size, args.code_size,
                args.dropout, args.dropword)
if not args.nocuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
best_loss = None

print("\nStart training")
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_ce, train_kld, kld_weight = train(corpus.train_data, model, optimizer, epoch)
        valid_ce, valid_kld = evaluate(corpus.valid_data, model)
        print('-' * 90)
        print("| epoch {:3d} | time {:4.2f}s | train loss {:4.4f} "
              "| train kld {:4.4f} | valid loss {:4.4f} | valid kld {:4.4f} "
              "| kld weight {:4.4f}".format(
                  epoch, time.time()-epoch_start_time, train_ce,
                  train_kld, valid_ce, valid_kld, kld_weight), flush=True)
        if best_loss is None or valid_ce + valid_kld < best_loss:
            best_loss = valid_ce + valid_kld
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                
except KeyboardInterrupt:
    print('-' * 90)
    print('Exiting from training early')


with open(args.save, 'rb') as f:
    model = torch.load(f)

test_ce, test_kld = evaluate(corpus.test_data, model)
print('=' * 90)
print("| End of training | test loss {:4.4f} | test kld {:4.4f}".format(
    test_ce, test_kld))
print('=' * 90)
