import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--kla', action='store_true',
                    help="do kl annealing")
parser.add_argument('--bow', action='store_true',
                    help="add bow loss during training")
args = parser.parse_args()


command_base = 'python main.py --data={0} --wd={1:.0e}{2}{3}'
dataset = args.data.rstrip('/').split('/')[-1]
logpath_base = 'logs/{0}/wd{1:.0e}{2}{3}.{0}.log'
lines = []
for wd in [0.001, 0.003, 0.01]:
    command = command_base.format(args.data, wd,
                                  ' --kla' if args.kla else '',
                                  ' --bow' if args.bow else '')

    logpath = logpath_base.format(dataset, wd,
                                  '.kla' if args.kla else '',
                                  '.bow' if args.bow else '')
    lines.append(command + ' > ' + logpath)

script_name = '{0}{1}{2}.sh'.format(dataset, '.kla' if args.kla else '',
                                    '.bow' if args.bow else '')
with open(script_name, 'w') as f:
    f.write('\n'.join(lines))
    
