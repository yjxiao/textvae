import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model import SSTextVAE
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data/yelp',
                    help="location of the data folder")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--y_embed_size', type=int, default=16,
                    help="size of the label embedding")
parser.add_argument('--hidden_size', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=16,
                    help="number of hidden units for RNN")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--num_labeled', type=int, default=1000,
                    help="number of labeled training samples")
parser.add_argument('--alpha', type=float, default=0.2,
                    help="weight given to the discriminative loss term")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--dropword', type=float, default=0,
                    help="dropout applied to input tokens (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--log_every', type=int, default=1000,
                    help="number of steps per logging event")
parser.add_argument('--bow', action='store_true',
                    help="add bag of words loss in training")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
parser.add_argument('--save', type=str,  default='./saves/model.pt',
                    help="path to save the final model")
args = parser.parse_args()

torch.manual_seed(args.seed)


def loss_function(targets, outputs, mu, logvar, y_logits, y_targets, bow):
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if y_targets is not None:
        class_nll_loss = F.nll_loss(y_logits, y_targets, size_average=False)
        kld_y = torch.tensor(0., device=ce_loss.device)
    else:
        class_nll_loss = torch.tensor(0., device=ce_loss.device)
        num_classes = y_logits.size(1)
        kld_y = torch.sum(y_logits.exp() * (y_logits + math.log(num_classes)))
    # bow: from batch_size x vocab_size to batch_size * sequence_length x vocab_size
    if not args.bow:
        bow_loss = torch.tensor(0., device=ce_loss.device)
    else:
        bow = bow.unsqueeze(1).expand(-1, outputs.size(1), -1).contiguous()
        bow_loss = F.cross_entropy(bow.view(bow.size(0) * bow.size(1),
                                            bow.size(2)),
                                   targets.view(-1),
                                   size_average=False,
                                   ignore_index=PAD_ID)
    return ce_loss, kld, class_nll_loss, kld_y, bow_loss


def evaluate(data_source, model):
    model.eval()
    total_ce = 0.0
    total_kld = 0.0
    total_bow = 0.0
    total_words = 0
    total_correct = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, targets, lengths, y_targets = data_source.get_batch(batch_size, i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_targets = y_targets.to(device)
        outputs, mu, logvar, bow, py = model(inputs, lengths, y=y_targets)
        ce, kld, _, _, bow_loss = loss_function(targets, outputs, mu, logvar,
                                                py, y_targets, bow)
        total_correct += (py.max(1)[1] == y_targets).sum().item()
        total_ce += ce.item()
        total_kld += kld.item()
        if args.bow:
            total_bow += bow_loss.item()
        total_words += sum(lengths)
    ppl = math.exp(total_ce / total_words)
    acc = total_correct / data_source.size
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_bow / data_source.size, ppl, acc)


def _train_step(data_source, model, optimizer, kld_weight, labeled, temperature):
    inputs, targets, lengths, y_targets = data_source.get_batch(args.batch_size, labeled=labeled)
    inputs = inputs.to(device)
    targets = targets.to(device)
    y_targets = y_targets.to(device) if labeled else None
    outputs, mu, logvar, bow, py = model(inputs, lengths, temperature, y_targets)
    ce, kld, nll_loss, kld_y, bow_loss = loss_function(targets, outputs, mu, logvar,
                                                       py, y_targets, bow)
    num_correct = (py.max(1)[1] == y_targets).sum().item() if labeled else 0
    optimizer.zero_grad()
    loss = ce + kld_weight * (kld + kld_y) + args.alpha * nll_loss + bow_loss
    loss.backward()
    optimizer.step()
    return ce.item(), kld.item(), kld_y.item(), bow_loss.item(), sum(lengths), num_correct

    
def train(data_source, model, optimizer, epoch):
    model.train()
    total_ce = 0.0
    total_kld = 0.0
    total_kld_y = 0.0
    total_bow = 0.0
    total_words = 0
    total_correct = 0
    temp = temperature_schedule((epoch - 1) * args.log_every)
    kld_weight = weight_schedule((epoch - 1) * args.log_every)
    l_steps, u_steps = allocate_steps(args.log_every, args.num_labeled, data_source.size)
    for steps, labeled in zip([l_steps, u_steps], [True, False]):
        for i in range(steps):
            ce, kld, kld_y, bow_loss, num_words, num_correct = _train_step(data_source, model,
                                                                           optimizer, kld_weight,
                                                                           labeled, temp)
            total_ce += ce
            total_kld += kld
            total_kld_y += kld_y
            total_bow += bow_loss
            total_words += num_words
            total_correct += num_correct
    ppl = math.exp(total_ce / total_words)
    num_samples = args.log_every * args.batch_size
    labeled_samples = l_steps * args.batch_size
    unlabeled_samples = u_steps * args.batch_size
    acc = total_correct / labeled_samples
    return (total_ce / num_samples, total_kld / num_samples, total_kld_y / unlabeled_samples,
            total_bow / num_samples, ppl, acc, kld_weight)


def interpolate(i, k, n):
    return max(min((i - k) / n, 1), 0)


def weight_schedule(t):
    """Scheduling of the KLD annealing weight. """
    return interpolate(t, 3000, 15000)


def temperature_schedule(t):
    return max(0.5, math.exp(-0.0001 * t))


def allocate_steps(total_steps, labeled_size, full_size):
    labeled_steps = int(total_steps * labeled_size / full_size)
    unlabeled_steps = total_steps - labeled_steps
    return labeled_steps, unlabeled_steps


print("Loading data")
corpus = data.SSCorpus(args.data, args.num_labeled, max_vocab_size=args.max_vocab,
                       max_length=args.max_length)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
print("Constructing model")
print(args)
device = torch.device('cpu' if args.nocuda else 'cuda')
model = SSTextVAE(vocab_size, corpus.num_classes, args.embed_size, args.y_embed_size,
                  args.hidden_size, args.code_size, args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
best_acc = 0

print("\nStart training")
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_ce, train_kld, train_kld_y, train_bow, train_ppl, train_acc, kld_weight = train(corpus.train_data, model, optimizer, epoch)
        valid_ce, valid_kld, valid_bow, valid_ppl, valid_acc = evaluate(corpus.valid_data, model)
        print('-' * 90)
        epoch_time = time.time()-epoch_start_time
        print("| epoch {:3d} | time {:4.2f}s | train loss {:4.4f} ({:4.4f}, {:4.4f}) "
              "| train ppl {:4.2f} | train acc {:2.2f} | kld weight {:4.2f}".format(
                  epoch, epoch_time, train_ce, train_kld, train_kld_y,
                  train_ppl, train_acc * 100, kld_weight))
        print("| epoch {:3d} | time {:4.2f}s | valid loss {:4.4f} ({:4.4f}, {:4.4f}) "
              "| valid ppl {:4.2f} | valid acc {:2.2f}".format(
                  epoch, epoch_time, valid_ce, valid_kld, 0,
                  valid_ppl, valid_acc * 100), flush=True)
        if valid_acc > best_acc:
            best_acc = valid_acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                
except KeyboardInterrupt:
    print('-' * 90)
    print('Exiting from training early')


with open(args.save, 'rb') as f:
    model = torch.load(f)

test_ce, test_kld, test_bow, test_ppl, test_acc = evaluate(corpus.test_data, model)
print('=' * 90)
print("| End of training | test loss {:4.4f} ({:4.4f}) | test bow loss {:4.4f} "
      "| test ppl {:2.2f} | test acc {:2.2f}".format(
          test_ce, test_kld, test_bow, test_ppl, test_acc * 100))
print('=' * 90)
