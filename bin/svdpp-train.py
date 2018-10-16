#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script generates the SVDPP model

After installation, you may run the following command:

$ svdpp-train [train-file]

This will generate a model file of the name [train-file]-svdpp-model.pck
'''

# Hung-Hsuan Chen <hhchen1105@gmail.com>
# Creation Date : 10-16-2018

import argparse
import pickle
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_file", type=str,
        help="Training file")
parser.add_argument("-lat", "--lmbda_latent", type=float, default=50,
        help="regularization term for latent factors")
parser.add_argument("-bias", "--lmbda_bias", type=float, default=0.01,
        help="regularization term for bias")
parser.add_argument("-impli", "--lmbda_impli", type=float, default=0.01,
        help="regularization term for bias")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.005,
        help="learning rate")
args = parser.parse_args()

import rdf


def check_args():
    if args.train_file is None or not os.path.isfile(args.train_file):
        print("train_file argument must be a valid file")
        parser.print_help()
        sys.exit(-1)


def svdpp_train():
    X = rdf.utils.load_data(args.train_file)
    n_users, n_items = rdf.utils.get_num_users_items(X)
    m = rdf.svdpp.SVDPP(
            n_users=n_users, n_items=n_items, lr=args.learning_rate,
            lmbda_latent=args.lmbda_latent, lmbda_bias=args.lmbda_bias,
            lmbda_impli=args.lmbda_impli)
    m.train(X)
    return m


def save_model(m):
    filename_prefix = os.path.splitext(os.path.basename(args.train_file))[0]
    with open('%s-svdpp-model.pck' % (filename_prefix), 'wb') as f_out:
        pickle.dump(m, f_out)


def main(argv):
    check_args()

    m = svdpp_train()
    save_model(m)


if __name__ == "__main__":
    main(sys.argv)
