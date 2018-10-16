======
Regularization differentiating function
======

This package implements the the method of "regularization differentiating function" (RDF).
In particular, we apply RDF on the following three recommendation modules: SVD, SVD++, and NMF.

****************************
Sample usage (with caution):
****************************

>>> import rdf
>>> X = rdf.utils.load_data('./data/filmtrust/train.dat')
>>> n_users, n_items = rdf.utils.get_num_users_items(X)
>>> m = rdf.rdfsvd.RDFSVD(
...   n_users=n_users, n_items=n_items, lr=.005,
...   lmbda_p=500, lmbda_q=500,
...   lmbda_u=.01, lmbda_i=.01,
...   method="linear")
>>> m.train(X)
>>> X_test = rdf.utils.load_data('./data/filmtrust/test.dat')
>>> X_pred = m.predict(X_test)
>>> rdf.utils.compute_rmse(X_test, X_pred)

***************
How to install:
***************

``python setup.py install``

*******************
Command line tools:
*******************

After installation, you may run the following scripts directly (tested in Ubuntu 16.04 and OS X El Capitan).

To generate the svd model with RDF, run:
========================================

``rdfsvd-train.py -tr [train-file] -m [model-file]``

This will generate a model file of the name ``[train-file]-rdfsvd-model.pck`` under the same directory.

The ``[test-file]`` contains a list of (user-id, item-id, rating), one tuple per line.

The ``[model-file]`` is the trained file.

