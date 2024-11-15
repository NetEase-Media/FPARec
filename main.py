import os
import time
import argparse
import math
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import print_seq_len_percentile, Timer, data_partition, exact_evaluate


def str2bool(s):
    if s not in {'False', 'True', '0', '1'}:
        raise ValueError('Not a valid boolean string')
    if s == 'True' or s == '1':
        return True
    else:
        return False


def str2ints(s):
    values = s.split(',')
    values = tuple(map(int, values))
    return values


def split_by_comma(s):
    return s.split(',')


def add_summary(summary_writer, tag, value, global_step):
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),
                               global_step=global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--max_steps', default=0, type=int)
    parser.add_argument('--eval_every', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--exact_evaluate', default=True, type=str2bool)
    parser.add_argument('--pred_batch_size_factor', default=2, type=float)
    parser.add_argument('--n_workers', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--adam_beta2', default=0.98, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--embedding_initializer', default='', type=str, choices=('', 'truncated_normal'))
    parser.add_argument('--embedding_scale', default=True, type=str2bool)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--positional_embedding', default=True, type=str2bool)
    parser.add_argument('--inner_size', default=50, type=int)
    parser.add_argument('--inner_act', default='relu', type=str)
    parser.add_argument('--inner_dropout', default=True, type=str2bool)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--context_dropout', default=False, type=str2bool)
    parser.add_argument('--pre_norm', default=True, type=str2bool)
    parser.add_argument('--post_norm', default=False, type=str2bool)
    parser.add_argument('--ffn', default=True, type=str2bool)
    parser.add_argument('--attention_type', default='dot_product', type=lambda x: x.split(','))
    parser.add_argument('--positional_attention_initializer', default='truncated_normal', type=str,
                        choices=('truncated_normal', 'zeros'))
    parser.add_argument('--factorized_positional_attention_k', default=20, type=int)
    parser.add_argument('--attention_temperature', default=1.0, type=float)
    parser.add_argument('--attention_kernel', default='relu', type=str, choices=('relu', 'elu'))
    parser.add_argument('--annealing_factor', default=1.0, type=float)
    parser.add_argument('--linear_projection_and_dropout', default=False, type=str2bool)
    parser.add_argument('--remove_masked_seq_emb', default=False, type=str2bool)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--training_target', default='all', type=str, choices=('all', 'last'))
    parser.add_argument('--loss_type', default='sparse_ce', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)

    args = parser.parse_args()

    train_dir = args.train_dir
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    with open(os.path.join(train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
    f.close()

    with Timer('Load data and preprocess data'):
        dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('Total number of interactions: {}'.format(cc))
    print('Average sequence length: %.2f' % (cc / len(user_train)))
    print_seq_len_percentile(user_train, message='Sequence length percentile: ')
    f = open(os.path.join(train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                          n_workers=args.n_workers)
    model = Model(usernum, itemnum, args)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    sess.run(tf.initialize_all_variables())
    T = 0.0
    t0 = time.time()
    num_batch = sampler.length / args.batch_size
    num_batch = math.ceil(num_batch)

    with Timer('Training and evaluation'):
        try:
            for epoch in range(1, args.num_epochs + 1):

                for step in tqdm(list(range(num_batch)), total=num_batch, ncols=70, leave=True, unit='b'):
                    u, seq, pos, neg = sampler.next_batch()
                    feed_dict = {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True, }
                    loss, _, summary, global_step = sess.run(
                        [model.loss, model.train_op, model.merged, model.global_step], feed_dict=feed_dict,
                    )
                    if step % 50 == 0:
                        summary_writer.add_summary(summary, global_step)
                    if args.max_steps > 0 and step > args.max_steps:
                        break
                if epoch % args.eval_every == 0:
                    t1 = time.time() - t0
                    T += t1

                    exact_t_test = exact_evaluate(model, dataset, args, sess,
                                                  batch_size=int(args.batch_size * args.pred_batch_size_factor),
                                                  evaluate_user=(), evaluate_item=())
                    exact_t_valid = exact_evaluate(model, dataset, args, sess, mode='valid',
                                                   batch_size=int(args.batch_size * args.pred_batch_size_factor),
                                                   evaluate_user=(), evaluate_item=())
                    print('')
                    print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                        epoch, T, exact_t_valid[0], exact_t_valid[1], exact_t_test[0], exact_t_test[1]))
                    add_summary(summary_writer, 'valid/NDCG10', exact_t_valid[0], global_step)
                    add_summary(summary_writer, 'valid/HR@10', exact_t_valid[1], global_step)
                    add_summary(summary_writer, 'test/NDCG10', exact_t_test[0], global_step)
                    add_summary(summary_writer, 'test/HR@10', exact_t_test[1], global_step)
                    if len(exact_t_valid) == 4 and len(exact_t_test) == 4:
                        print('epoch:%d, time: %f(s), valid (NDCG@100: %.4f, HR@100: %.4f), test (NDCG@100: %.4f, HR@100: %.4f)' % (
                            epoch, T, exact_t_valid[2], exact_t_valid[3], exact_t_test[2], exact_t_test[3]))
                        add_summary(summary_writer, 'valid/NDCG100', exact_t_valid[2], global_step)
                        add_summary(summary_writer, 'valid/HR@100', exact_t_valid[3], global_step)
                        add_summary(summary_writer, 'test/NDCG100', exact_t_test[2], global_step)
                        add_summary(summary_writer, 'test/HR@100', exact_t_test[3], global_step)

                    t0 = time.time()
        except:
            import traceback
            traceback.print_exc()
            import sys
            sampler.close()
            f.close()
            exit(1)

    f.close()
    sampler.close()
    print("Done training")


if __name__ == '__main__':
    main()
