import collections

def get_num_users_items(X):
    users = set()
    items = set()
    for (user, item, r) in X:
        users.add(user)
        items.add(item)
    return len(users), len(items)


def load_data(filename, sep="\t"):
    X = []
    with open(filename) as f:
        for line in f:
            user, item, rating = line.strip().split(sep)
            X.append((user,item, float(rating)))
    return X


def get_all_num_user_ratings(X):
    n_user_ratings = collections.defaultdict(int)
    for (user, item, r) in X:
        n_user_ratings[user] += 1
    return sorted(n_user_ratings.values())


def get_all_num_item_received_ratings(X):
    n_item_received_ratings= collections.defaultdict(int)
    for (user, item, r) in X:
        n_item_received_ratings[item] += 1
    return sorted(n_item_received_ratings.values())


def compute_rmse(X_test, X_pred):
    sse = 0.
    for i in range(len(X_test)):
        u_test, i_test, r_test = X_test[i]
        u_pred, i_pred, r_pred = X_pred[i]
        assert(u_test == u_pred)
        assert(i_test == i_pred)
        sse += (r_test - r_pred) ** 2
    return (sse / len(X_test)) ** .5
