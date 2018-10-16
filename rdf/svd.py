# Hung-Hsuan Chen <hhchen1105@gmail.com>
# Creation Date : 09-02-2017
# Last Modified: Tue 20 Mar 2018 01:16:50 PM CST

import numpy as np


class SVD():
    '''
    the parameters are adopted from recommender systems handbook section 5.3.1.
    (http://www.cs.ubbcluj.ro/~gabis/DocDiplome/SistemeDeRecomandare/Recommender_systems_handbook.pdf)
    '''
    def __init__(
            self, n_users, n_items, n_factors=15, n_epochs=50,
            lr=.005, lr_bias=None, lr_latent=None,
            lmbda=.02, lmbda_bias=None, lmbda_latent=None, lr_shrink_rate=.9):
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
        self.P = np.random.randn(self.n_users, self.n_factors)
        self.Q = np.random.randn(self.n_items, self.n_factors)
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)

    def train(self, ratings, validate_ratings=None, show_process_rmse=True):
        self._external_internal_id_mapping(ratings)
        self.global_mean = self._compute_global_mean(ratings)

        for epoch in range(self.n_epochs):
            epoch_shrink = self.lr_shrink_rate ** epoch
            for (ext_user_id, ext_item_id, r) in ratings:
                err = r - self.predict_single_rating(ext_user_id, ext_item_id)
                u = self.eu2iu[ext_user_id]
                i = self.ei2ii[ext_item_id]
                r = float(r)
                bu = self.bu[u]
                bi = self.bi[i]
                pu = self.P[u,:]
                qi = self.Q[i,:]
                self.bu[u] -= (self.lr_bias * epoch_shrink) * (-err + self.lmbda_bias * bu)
                self.bi[i] -= (self.lr_bias * epoch_shrink) * (-err + self.lmbda_bias * bi)
                self.P[u,:] -= (self.lr_latent * epoch_shrink) * (-err * qi + self.lmbda_latent * pu)
                self.Q[i,:] -= (self.lr_latent * epoch_shrink) * (-err * pu + self.lmbda_latent * qi)
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
