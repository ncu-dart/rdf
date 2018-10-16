# Hung-Hsuan Chen <hhchen1105@gmail.com>
# Creation Date : 09-02-2017
# Last Modified: Sat 28 Apr 2018 09:02:20 AM CST

import collections
import numpy as np


class RDFNMF():
    # XXX: lr_shrink_rate is not used here... how to put it to the equation?
    def __init__(
            self, n_users, n_items, n_factors=15, n_epochs=50,
            lr=.005, lr_bias=None, lr_latent=None,
            lmbda=.02, lmbda_bias=None, lmbda_latent=None,
            lr_shrink_rate=.9, method="log", alpha=np.exp(1)):
        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr = lr
        self.lr_bias = lr if lr_bias is None else lr_bias
        self.lr_latent = lr if lr_latent is None else lr_latent
        self.lmbda = lmbda
        self.lmbda_bias = lmbda if lmbda_bias is None else lmbda_bias
        self.lmbda_latent = lmbda if lmbda_latent is None else lmbda_latent
        self.lr_shrink_rate = lr_shrink_rate
        self.eu2iu = {} # external-user to internal-user id
        self.iu2eu = {} # internal-user to external-user id
        self.ei2ii = {} # external-item to internal-item id
        self.ii2ei = {} # internal-item to external-item id
        self.P = np.random.random([self.n_users, self.n_factors]) # rand float in [0, 1.)
        self.Q = np.random.random([self.n_items, self.n_factors]) # rand float in [0, 1.)
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
            user_num = np.zeros((self.n_users, self.n_factors))
            user_denom = np.zeros((self.n_users, self.n_factors))
            item_num = np.zeros((self.n_items, self.n_factors))
            item_denom = np.zeros((self.n_items, self.n_factors))

            for (ext_user_id, ext_item_id, r) in ratings:
                r_hat = self.predict_single_rating(ext_user_id, ext_item_id)
                err = r - r_hat
                u = self.eu2iu[ext_user_id]
                i = self.ei2ii[ext_item_id]
                r = float(r)
                bu = self.bu[u]
                bi = self.bi[i]
                pu = self.P[u, :]
                qi = self.Q[i, :]
                reg_bias = (
                    self.lmbda_bias / (self.n_item_rating[i] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_bias / np.sqrt(self.n_item_rating[i] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_bias / np.log(self.n_item_rating[i] + self.alpha)
                    )
                self.bu[u] -= (self.lr_bias * epoch_shrink) * (-err + reg_bias * bu)
                self.bi[i] -= (self.lr_bias * epoch_shrink) * (-err + reg_bias * bi)

                user_num[u, :] += qi * r
                user_denom[u, :] += qi * r_hat
                item_num[i, :] += pu * r
                item_denom[i, :] += pu * r_hat

            # update user's latent factors
            reg_latent = (
                self.lmbda_latent / (self.n_user_rating[u] + self.alpha)
                ) if self.method == "linear" else (
                self.lmbda_latent / np.sqrt(self.n_user_rating[u] + self.alpha)
                ) if self.method == "sqrt" else (
                self.lmbda_latent / np.log(self.n_user_rating[u] + self.alpha)
                )
            for u in range(self.n_users):
                user_denom[u, :] += self.n_user_rating[u] * reg_latent * self.P[u, :]
                self.P[u, :] *= user_num[u, :] / user_denom[u, :]
            for i in range(self.n_items):
                item_denom[i, :] += self.n_item_rating[i] * reg_latent * self.Q[i, :]
                self.Q[i, :] *= item_num[i, :] / item_denom[i, :]

            if show_process_rmse:
                if validate_ratings is None:
                    loss, rmse = self._compute_err(ratings)
                    print("After %i epochs, loss=%.6f, training rmse=%.6f" % (epoch+1, loss, rmse))
                else:
                    loss, rmse = self._compute_err(validate_ratings)
                    print("After %i epochs, loss=%.6f, validating rmse=%.6f" % (epoch+1, loss, rmse))
            else:
                print("After %i epoch" % (epoch+1))

    def predict_single_rating(self, ext_user_id, ext_item_id):
        u = self.eu2iu[ext_user_id] if ext_user_id in self.eu2iu else -1
        i = self.ei2ii[ext_item_id] if ext_item_id in self.ei2ii else -1
        bu = self.bu[u] if u >= 0 else 0
        bi = self.bi[i] if i >= 0 else 0
        pu = self.P[u,:] if u >= 0 else np.zeros(self.n_factors)
        qi = self.Q[i,:] if i >= 0 else np.zeros(self.n_factors)
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

    def _gen_n_user_rated_items(self, ratings):
        user_rated_items = collections.defaultdict(set)
        for (eu, ei, r) in ratings:
            user_rated_items[self.eu2iu[eu]].add(self.ei2ii[ei])

        n_user_rated_items = {}
        for u in user_rated_items:
            n_user_rated_items[u] = len(user_rated_items[u])
        return n_user_rated_items

    def _gen_n_item_received_ratings(self, ratings):
        item_received_ratings = collections.defaultdict(set)
        for (eu, ei, r) in ratings:
            item_received_ratings[self.ei2ii[ei]].add(self.eu2iu[eu])

        n_item_received_ratings = {}
        for i in item_received_ratings:
            n_item_received_ratings[i] = len(item_received_ratings[i])
        return n_item_received_ratings

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
        loss = 0.
        sse = 0.

        for (ext_user_id, ext_item_id, r) in ratings:
            #u = self.eu2iu[ext_user_id] if ext_user_id in self.eu2iu else -1
            #i = self.ei2ii[ext_item_id] if ext_item_id in self.ei2ii else -1
            r = float(r)
            err_square = (r - self.predict_single_rating(ext_user_id, ext_item_id)) ** 2
            sse += err_square
            loss += err_square
        loss += self.lmbda_latent * (np.linalg.norm(self.P) + np.linalg.norm(self.Q)) + \
                self.lmbda_bias * (np.linalg.norm(self.bu) + np.linalg.norm(self.bi))
        return loss, (sse / len(ratings)) ** .5
