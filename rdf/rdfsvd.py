# Hung-Hsuan Chen <hhchen1105@gmail.com>
# Creation Date : 09-02-2017
# Last Modified: Sat 28 Apr 2018 08:51:31 AM CST

import collections


import numpy as np


class RDFSVD():
    """
    SVD with Regularization Differentiating Function
    """
    def __init__(
            self, n_users, n_items, n_factors=15, n_epochs=50,
            lr=.005, lr_bias=None, lr_latent=None,
            lmbda=1., lmbda_p=None, lmbda_q=None, lmbda_u=None, lmbda_i=None,
            lr_shrink_rate=.9, method="log", alpha=np.exp(1)):
        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr = lr
        self.lr_bias = lr if lr_bias is None else lr_bias
        self.lr_latent = lr if lr_latent is None else lr_latent
        self.lmbda = lmbda
        self.lmbda_p = lmbda if lmbda_p is None else lmbda_p
        self.lmbda_q = lmbda if lmbda_q is None else lmbda_q
        self.lmbda_u = lmbda if lmbda_u is None else lmbda_u
        self.lmbda_i = lmbda if lmbda_i is None else lmbda_i
        self.lr_shrink_rate = lr_shrink_rate
        self.eu2iu = {}  # external-user to internal-user id
        self.iu2eu = {}  # internal-user to external-user id
        self.ei2ii = {}  # external-item to internal-item id
        self.ii2ei = {}  # internal-item to external-item id
        self.P = np.random.randn(self.n_users, self.n_factors)
        self.Q = np.random.randn(self.n_items, self.n_factors)
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.method = method
        self.alpha = alpha
        self.n_user_rating = None
        self.n_item_rating = None

    def train(self, ratings, validate_ratings=None, show_process_rmse=True):
        self._external_internal_id_mapping(ratings)
        self.global_mean = self._compute_global_mean(ratings)
        self._compute_n_user_item_rating(ratings)

        for epoch in range(self.n_epochs):
            epoch_shrink = self.lr_shrink_rate ** epoch
            for (ext_user_id, ext_item_id, r) in ratings:
                err = r - self.predict_single_rating(ext_user_id, ext_item_id)
                u = self.eu2iu[ext_user_id]
                i = self.ei2ii[ext_item_id]
                r = float(r)
                bu = self.bu[u]
                bi = self.bi[i]
                pu = self.P[u, :]
                qi = self.Q[i, :]

                reg_bu = (
                    self.lmbda_u / (self.n_user_rating[u] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_u / np.sqrt(self.n_user_rating[u] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_u / np.log(self.n_user_rating[u] + self.alpha)
                    )
                reg_bi = (
                    self.lmbda_i / (self.n_item_rating[i] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_i / np.sqrt(self.n_item_rating[i] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_i / np.log(self.n_item_rating[i] + self.alpha)
                    )
                reg_pu = (
                    self.lmbda_p / (self.n_user_rating[u] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_p / np.sqrt(self.n_user_rating[u] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_p / np.log(self.n_user_rating[u] + self.alpha)
                    )
                reg_qi = (
                    self.lmbda_q / (self.n_item_rating[i] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_q / np.sqrt(self.n_item_rating[i] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_q / np.log(self.n_item_rating[i] + self.alpha)
                    )

                self.bu[u] -= (self.lr_bias * epoch_shrink) * (
                    -err + reg_bu * bu)
                self.bi[i] -= (self.lr_bias * epoch_shrink) * (
                    -err + reg_bi * bi)
                self.P[u, :] -= (self.lr_latent * epoch_shrink) * (
                    -err * qi + reg_pu * pu)
                self.Q[i, :] -= (self.lr_latent * epoch_shrink) * (
                    -err * pu + reg_qi * qi)
            if show_process_rmse:
                if validate_ratings is None:
                    rmse = self._compute_err(ratings)
                    print("After %i epochs, training rmse=%.6f" % (
                        epoch+1, rmse))
                else:
                    rmse = self._compute_err(validate_ratings)
                    print("After %i epochs, validating rmse=%.6f" % (
                        epoch+1, rmse))
            else:
                print("After %i epoch" % (epoch+1))

    def predict_single_rating(self, ext_user_id, ext_item_id):
        u = self.eu2iu[ext_user_id] if ext_user_id in self.eu2iu else -1
        i = self.ei2ii[ext_item_id] if ext_item_id in self.ei2ii else -1
        bu = self.bu[u] if u >= 0 else 0
        bi = self.bi[i] if i >= 0 else 0
        pu = self.P[u, :] if u >= 0 else np.zeros(self.n_factors)
        qi = self.Q[i, :] if i >= 0 else np.zeros(self.n_factors)
        return self.global_mean + bu + bi + np.dot(pu, qi)

    def predict(self, user_item_pairs):
        all_predicts = []
        for (ext_user_id, ext_item_id, r) in user_item_pairs:
            all_predicts.append((ext_user_id, ext_item_id, self.predict_single_rating(ext_user_id, ext_item_id)))
        return all_predicts

    def _external_internal_id_mapping(self, ratings):
        for (eu, ei, r) in ratings:
            if eu not in self.eu2iu:
                iu = len(self.eu2iu)
                self.eu2iu[eu] = iu
                self.iu2eu[iu] = eu
            if ei not in self.ei2ii:
                ii = len(self.ei2ii)
                self.ei2ii[ei] = ii
                self.ii2ei[ii] = ei

    def _compute_global_mean(self, ratings):
        rating_sum = 0.
        for ext_user_id, ext_item_id, r in ratings:
            rating_sum += float(r)
        return rating_sum / len(ratings)

    def _compute_n_user_item_rating(self, ratings):
        n_user_rating = collections.defaultdict(int)
        n_item_rating = collections.defaultdict(int)
        for (ext_user_id, ext_item_id, r) in ratings:
            u = self.eu2iu[ext_user_id]
            i = self.ei2ii[ext_item_id]
            n_user_rating[u] += 1
            n_item_rating[i] += 1
        self.n_user_rating = dict(n_user_rating)
        self.n_item_rating = dict(n_item_rating)

    def _compute_err(self, ratings):
        sse = 0.

        for (ext_user_id, ext_item_id, r) in ratings:
            r = float(r)
            err_square = (
                r - self.predict_single_rating(ext_user_id, ext_item_id)) ** 2
            sse += err_square
        return (sse / len(ratings)) ** .5
