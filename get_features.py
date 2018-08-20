import argparse
import os
import torch
import numpy as np
import data

parser = argparse.ArgumentParser(description='Text VAE translate')
parser.add_argument('--data', type=str, default='./data/yahoo',
                    help='location of the corpus (same as training)')
parser.add_argument('--ckpt', type=str, default='./saves/model.pt',
                    help='location of the model file')
parser.add_argument('--output_dir', type=str, default='feats',
                    help='output directory to store extracted features')
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=20,
                    help='maximum generation length')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size used in generation')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)


def get_features(data_source, model, device):
    feats = []
    labs = []
    model.eval()
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, labels, lengths, _ = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        targets = texts[:, 1:].clone().to(device)
        labels = labels.to(device)
        outputs, mu, logvar, _ = model(inputs, labels, lengths)
        feats.append(mu[1].squeeze(0).detach().cpu().numpy())
        labs.append(labels.cpu().numpy())
    return np.concatenate(feats), np.concatenate(labs)

    
def main(args):
    with open(args.ckpt, 'rb') as f:
        model = torch.load(f)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    dataset = args.data.rstrip('/').split('/')[-1]
    with_label = True if dataset in ['yahoo', 'yelp'] else False
    print("Loading data")
    corpus = data.Corpus(
        args.data, max_vocab_size=args.max_vocab,
        max_length=args.max_length, with_label=with_label)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train.size)
    print("\tvocabulary size: ", vocab_size)
    filenames = ['{0}.{1}'.format(os.path.basename(args.ckpt), x) for x in ['train.pkl', 'valid.pkl', 'test.pkl']]
    results = [get_features(x, model, device) for x in [corpus.train, corpus.valid, corpus.test]]
    savepaths = [os.path.join(args.output_dir, dataset, x) for x in filenames]
    for result, savepath in zip(results, savepaths):
        with open(savepath, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main(args)
