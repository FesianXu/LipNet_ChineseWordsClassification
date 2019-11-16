# !/usr/bin/env python
# -*- coding:utf-8 -*-

import yaml
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='LipReadingNet')
    parser.add_argument('--work-dir',
        default=None,
        help='the model working directory')
    parser.add_argument(
        '--config',
        help='path to the configuration file')
    # workdir save the model_weights and the log files
    

    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--phase',
        type=str)
    parser.add_argument('--feeder',
        type=str)
    parser.add_argument('--train-feeder-args',
        type=dict,
        default=None)
    parser.add_argument('--test-feeder-args',
        type=dict,
        default=None)
    parser.add_argument('--eval-feeder-args',
        type=dict,
        default=None)
    parser.add_argument('--whole-train-feeder-args',
        type=dict,
        default=None)
    parser.add_argument('--trade-off-args',
        type=dict,
        default=None)
    parser.add_argument('--model-args',
        type=dict,
        default=None)
    

    parser.add_argument('--weight-decay',
        type=float)
    parser.add_argument('--base-lr',
        type=float)
    parser.add_argument('--step',
        type=int)
    parser.add_argument('--optimizer',
        type=str)   
    parser.add_argument('--train-batchsize',
        type=int)
    parser.add_argument('--eval-batchsize',
        type=int)
    parser.add_argument('--test-batchsize',
        type=int)
    parser.add_argument('--nesterov',
        type=str2bool)

    parser.add_argument('--weights',
        default=None)
    parser.add_argument('--log-interval',
        type=int,
        default=200,
        help='the interval for print the log')
    parser.add_argument('--eval-interval',
        type=int,
        default=5)
    parser.add_argument('--save-interval',
        type=int)
    parser.add_argument('--save-weight-strategy',
        type=str)
    parser.add_argument('--best-save-theshold',
        type=float)
    parser.add_argument('--is-detail-log',
        type=str2bool)
    parser.add_argument('--is-print-log',
        type=str2bool,
        default=True)
    parser.add_argument('--num-worker',
        type=int)
    parser.add_argument('--devices',
        type=list)
    parser.add_argument('--img-height',
        type=int)
    parser.add_argument('--img-width',
        type=int)
    parser.add_argument('--img-max-seq',
        type=int)
    parser.add_argument('--start-epoch',
        type=int)
    parser.add_argument('--num-epoch',
        type=int)

    return parser

def load_args():
    '''
    load the args from yaml files or the cmd
    '''
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
            parser.set_defaults(**default_args)
    arg = parser.parse_args()
    return arg

if __name__ == '__main__':
    load_args()