#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script shows the predictions of the SVD model

After installation, you may run the following command:

$ rdf-test.py [test-file] [model-file]

The [test-file] contains a list of query items, one behavior per line.
'''

import argparse
import os
import pickle
import sys

import rdf


parser = argparse.ArgumentParser()
parser.add_argument("-te", "--test_file", type=str,
        help="Test file")
parser.add_argument("-m", "--model_file", type=str,
        help="Model file")
args = parser.parse_args()


def check_args():
    if args.test_file is None or not os.path.isfile(args.test_file):
        print("test_file argument must be a valid file")
        parser.print_help()
        sys.exit(-1)

    if args.model_file is None or not os.path.isfile(args.model_file):
        print("model_file argument must be a valid file")
        parser.print_help()
        sys.exit(-1)


def load_model(filename):
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m


def main(argv):
    check_args()

    m = load_model(args.model_file)
    X_test = rdf.utils.load_data(args.test_file)
    X_pred = m.predict(X_test)
    rmse = rdf.utils.compute_rmse(X_test, X_pred)
    print("RMSE: %f" % (rmse))


if __name__ == "__main__":
    main(sys.argv)
