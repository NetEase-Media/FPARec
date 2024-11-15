import math
import sys
import copy
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time
from tabulate import tabulate


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(self.name, time.time() - self.start))
        return exc_type is None


def array_like(ref):
    return list()


def train_val_test_partition(User):
    user_train = {}
    user_valid = {}
    user_test = {}
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # user_train[user] = np.array(User[user], dtype=np.int32)
            user_train[user] = User[user]
            user_valid[user] = array_like(User[user])
            user_test[user] = array_like(User[user])
        else:
            # user_train[user] = np.array(User[user][:-2], dtype=np.int32)
            user_train[user] = User[user][:-2]
            user_valid[user] = array_like(User[user])
            user_valid[user].append(User[user][-2])
            user_test[user] = array_like(User[user])
            user_test[user].append(User[user][-1])
    return user_train, user_valid, user_test


def print_seq_len_percentile(seqs, message):
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile([len(v) for k, v in seqs.items()], percentiles)
    print(message)
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    print('user_num: {}'.format(usernum))
    print('item_num: {}'.format(itemnum))
    print('Total number of interactions in train + val + test: {}'.format(sum([len(v) for v in User.values()])))
    print('Average sequence length of train + val + test: {:.2f}'.format(sum([len(v) for v in User.values()]) / len(User.values())))
    print_seq_len_percentile(User, message='Sequence length percentile in train + val + test: ')

    user_train, user_valid, user_test = train_val_test_partition(User)
    return [user_train, user_valid, user_test, usernum, itemnum]


def print_item_frequency_percentile(frequencies):
    percentiles = [10, 25, 50, 75, 90]
    values = np.percentile(frequencies, percentiles)
    print('Item frequency percentile: ')
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))


def print_percentile(values):
    percentiles = [10, 25, 50, 75, 90]
    values = np.percentile(values, percentiles)
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def build_seq(train, valid, test, u, args):
    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    if test is not None:
        seq[idx] = test[u][0]
        idx -= 1

    if valid is not None:
        seq[idx] = valid[u][0]
        idx -= 1

    for i in reversed(train[u]):
        seq[idx] = i
        idx -= 1
        if idx == -1: break

    return seq


def exact_evaluate(model, dataset, args, sess, mode='test', batch_size=0, sample_user_num=10000,
                   evaluate_user=(), evaluate_item=()):
    train, valid, test, user_num, item_num = dataset
    valid_user = 0.0

    users = list(evaluate_user) if len(evaluate_user) > 0 else range(1, user_num + 1)
    if sample_user_num > 0 and len(users) > sample_user_num:
        users = random.sample(users, sample_user_num)
        eval_user_num = sample_user_num
    else:
        eval_user_num = len(users)

    if batch_size > 0:
        us = []
        seqs = []
        target_item_ids = []
        print('Preparing evaluate data...')
        for u in tqdm(users, total=eval_user_num, leave=False, ncols=70):
            if len(train[u]) < 1:
                continue
            if mode == 'valid' and len(valid[u]) < 1:
                continue
            if mode == 'test' and len(test[u]) < 1:
                continue

            seq = build_seq(train, valid if mode is 'test' else None, None, u, args)

            # Put target_item_idx in 0-position
            target_item_id = valid[u][0] if mode == 'valid' else test[u][0]

            us.append(u)
            seqs.append(seq)
            target_item_ids.append(target_item_id)
            valid_user += 1
        print('Predicting on evaluate data...')
        top_k_values, top_k_indices = model.predict_top_k(sess, us, seqs, list(range(1, item_num + 1)),
                                                          batch_size=batch_size)
        top_k_indices += 1

        top_10_indices = top_k_indices[:, :10]
        HT = np.sum(np.expand_dims(target_item_ids, axis=-1) == top_10_indices)
        NDCG = np.sum(
            (np.expand_dims(target_item_ids, axis=-1) == top_10_indices) *
            (1 / np.expand_dims(np.log2(np.arange(0, 10) + 2), axis=0))
        )
        HT /= valid_user
        NDCG /= valid_user

        top_100_indices = top_k_indices[:, :100]
        HT100 = np.sum(np.expand_dims(target_item_ids, axis=-1) == top_100_indices)
        NDCG100 = np.sum(
            (np.expand_dims(target_item_ids, axis=-1) == top_100_indices) *
            (1 / np.expand_dims(np.log2(np.arange(0, 100) + 2), axis=0))
        )
        HT100 /= valid_user
        NDCG100 /= valid_user
        return NDCG, HT, NDCG100, HT100
    else:
        raise ValueError