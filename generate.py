import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable

import data
from data import SOS_ID, EOS_ID

parser = argparse.ArgumentParser(description='Text VAE translate')
parser.add_argument('--data', type=str, default='./data/books',
                    help='location of the corpus (same as training)')
parser.add_argument('--ckpt', type=str, default='./saves/model.pt',
                    help='location of the model file')
parser.add_argument('--task', type=str, default='get_code',
                    help='task to perform [sample, reconstruct, get_code]')
parser.add_argument('--input_file', type=str, default='test.txt',
                    help='location of the input texts')
parser.add_argument('--output_file', type=str, default='outputs.txt',
                    help='output file to write reconstructed texts')
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=20,
                    help='maximum generation length')
parser.add_argument('--num_samples', type=int, default=1,
                    help='number of samples per reconstruction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size used in generation')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)


def indices_to_sentence(indices, idx2word):
    words = []
    for i in indices:
        if i == EOS_ID:
            break
        else:
            words.append(idx2word[i])
    return ' '.join(words)


def sample(data_source, model, label, idx2word, device):
    results = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.num_samples - i, args.batch_size)
        if data_source.has_label:
            labels = torch.full((batch_size,), label, dtype=torch.long, device=device)
            samples = model.sample(labels, args.max_length, SOS_ID)
        else:
            samples = model.sample(batch_size, args.max_length, SOS_ID, device)
        for i, sample in enumerate(samples.cpu().numpy()):
            if data_source.has_label:
                label = labels[i].item()
                prefix = '{0:d}\t'.format(label)
            else:
                prefix = ''
            results.append(prefix + indices_to_sentence(sample, idx2word))
    return results


def reconstruct(data_source, model, idx2word, device):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, labels, lengths, idx = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        if data_source.has_label:
            labels = labels.to(device)
            samples = model.reconstruct(inputs, labels, lengths, args.max_length, SOS_ID)
        else:
            samples = model.reconstruct(inputs, lengths, args.max_length, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results


def get_z(data_source, model, device):
    mus, var = [], []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, _, lengths, idx = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        _, mu, logvar, _ = model.forward(inputs, lengths)
        for x in mu.squeeze(0).cpu().detach().numpy()[idx]:
            mus.append(' '.join(list(map(str, x))))
        for x in logvar.squeeze(0).cpu().detach().numpy()[idx]:            
            var.append(' '.join(list(map(str, np.exp(x)))))
    return mus, var


def main(args):
    with open(args.ckpt, 'rb') as f:
        model = torch.load(f)
    model.eval()
    device = torch.device('cuda' if args.cuda else 'cpu')
    model.to(device)
    dataset = args.data.rstrip('/').split('/')[-1]
    with_label = True if dataset in ['yahoo', 'yelp'] else False
    print("Loading data")
    corpus = data.Corpus(args.data, max_vocab_size=args.max_vocab,
                         max_length=args.max_length, with_label=with_label)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train.size)
    print("\tvocabulary size: ", vocab_size)
    # data to be reconstructed
    input_path = os.path.join(args.data, args.input_file)
    output_path = os.path.join(args.data, args.output_file)
    input_data = data.Data(input_path, (corpus.word2idx, corpus.label2idx), with_label=with_label)
    if args.task == 'sample':
        if with_label:
            for label in range(corpus.num_classes):
                results = sample(input_data, model, label, corpus.idx2word, device)
                with open('{0}.{1:d}.samp'.format(output_path, label), 'w') as f:
                    f.write('\n'.join(results))
        else:
            results = sample(input_data, model, None, corpus.idx2word, device)
            with open('{0}.samp'.format(output_path), 'w') as f:
                f.write('\n'.join(results))

    elif args.task == 'reconstruct':
        for i in range(args.num_samples):
            results = reconstruct(input_data, model, corpus.idx2word, device)
            with open('{0}.{1:d}.rec'.format(output_path, i), 'w') as f:
                f.write('\n'.join(results))

    elif args.task == 'get_code':
        mus, var = get_z(input_data, model, device)
        with open('{0}.mu'.format(output_path), 'w') as f:
            f.write('\n'.join(mus))
        with open('{0}.var'.format(output_path), 'w') as f:
            f.write('\n'.join(var))

            
if __name__ == '__main__':
    main(args)
