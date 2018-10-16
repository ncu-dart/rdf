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

To generate the SVD model with RDF, run:
========================================

``rdfsvd-train.py -tr [train-file] -m [model-file]``

This will run SVD with RDF and generate a model file of the name ``[train-file]-rdfsvd-model.pck`` under the same directory.

Similarly, you may use the command ``rdfsvdpp-train.py``, ``rdfnmf-train.py`` to generate the SVD++ model with RDF and NMF model with RDF.

We also include the SVD model, SVD++ model, and NMF model without RDF.  You may execute these models by running ``svd-train.py``, ``svdpp-train.py`, and ``nmf-train.py``.

To test the model and report rmse score, run:
========================================

``rdf-test.py -te [test-file] -m [model-file]``

The ``[test-file]`` contains a list of (user-id, item-id, rating), one tuple per line.

The ``[model-file]`` is the trained file.

