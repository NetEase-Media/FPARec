from modules import *
from tqdm import tqdm
import math


class BatchInputAndConcatResult(object):
    def __init__(self, data, batch_size):
        self.data = data
        self.max_i = len(data[0])
        if batch_size == 0:
            self.batch_size = self.max_i
        else:
            self.batch_size = batch_size
        self.i = 0
        self.result = []
        self.result_tuple_size = 0

    def __iter__(self):
        self.i = 0
        self.result = []
        return self

    def __next__(self):
        if self.i < self.max_i:
            i = self.i
            self.i = min(i + self.batch_size, self.max_i)
            return [d[i:self.i] for d in self.data]
        else:
            raise StopIteration

    def __len__(self):
        return math.ceil(self.max_i / self.batch_size)

    def update_result(self, batch_result):
        if isinstance(batch_result, tuple):
            self.result_tuple_size = len(batch_result)
            if len(self.result) == 0:
                self.result = [[] for i in range(self.result_tuple_size)]
            for i in range(self.result_tuple_size):
                self.result[i].append(batch_result[i])
        else:
            self.result.append(batch_result)

    def concat_result(self):
        if self.result_tuple_size > 0:
            tuple_size = len(self.result)
            self.result = tuple([np.concatenate(self.result[i], axis=0) for i in range(tuple_size)])
        else:
            self.result = np.concatenate(self.result, axis=0)
        return self.result


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=(), name='input_is_training')
        self.u = tf.placeholder(tf.int32, shape=(None, ), name='input_u')
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_seq')
        self.input_interval = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_neg')
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 initializer=args.embedding_initializer,
                                                 scale=args.embedding_scale,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0),
                        [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                initializer=args.embedding_initializer,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

                # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq) if args.pre_norm else self.seq,
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   attention_type=args.attention_type,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   linear_projection_and_dropout=args.linear_projection_and_dropout,
                                                   args=args,
                                                   scope="self_attention")
            if args.post_norm:
                self.seq = normalize(self.seq)
            elif args.pre_norm:
                self.seq = normalize(self.seq)
            # Feed forward
            self.seq = feedforward(self.seq,
                                   num_units=[args.inner_size, args.hidden_units],
                                   inner_act=args.inner_act,
                                   dropout_rate=args.dropout_rate,
                                   inner_dropout=args.inner_dropout,
                                   is_training=self.is_training)
            self.seq *= mask
            if args.post_norm:
                self.seq = normalize(self.seq)

            if args.pre_norm:
                self.seq = normalize(self.seq)  # (N, T, C)

            if args.training_target == 'last':
                pos = pos[:, -1]  # (N, )
                if args.sampler_num_neg == 0:
                    neg = None
                elif args.sampler_num_neg > 1:
                    neg = neg[:, -args.sampler_num_neg:]
                    neg = tf.reshape(neg, (-1, ))  # (N * num_neg, )
                else:
                    neg = neg[:, -1]  # (N, )
                seq_emb = tf.reshape(self.seq[:, -1, :], [tf.shape(self.input_seq)[0], args.hidden_units])  # (N, C)
                # ignore padding items (0)
                istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0]])  # (N, )
            else:
                pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])  # (N, )
                neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])  # (N, )
                seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])  # (N*T, C)
                # ignore padding items (0)
                istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])

            self.test_item = tf.placeholder(tf.int32, shape=(None, ), name='input_test_item')
            self.test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)  # (M, C)

        ######## Test Graph Prediction Layer ########
        self.test_logits = tf.matmul(self.seq[:, -1, :], tf.transpose(self.test_item_emb))  # (N, M)
        self.test_top_k = tf.nn.top_k(self.test_logits, k=100)

        ######## Loss and Optimizer ########
        if args.loss_type == 'sparse_ce':
            self.loss = self.sparse_ce_loss(pos, item_emb_table, None, seq_emb, istarget)

        tf.summary.scalar('loss', self.loss)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_losses = sum(reg_losses)
        tf.summary.scalar('reg_loss', reg_losses)
        self.loss += reg_losses
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.histogram('item_emb_norm', tf.norm(item_emb_table, axis=-1))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=args.adam_beta2)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, batch_size=0):
        if batch_size == 0:
            return sess.run(self.test_logits,
                            {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
        else:
            batch_inputs = BatchInputAndConcatResult([u, seq], batch_size=batch_size)
            for batch_u, batch_seq in batch_inputs:
                batch_inputs.update_result(
                    sess.run(self.test_logits,
                             {self.u: batch_u,
                              self.input_seq: batch_seq,
                              self.test_item: item_idx,
                              self.is_training: False})
                )
            return batch_inputs.concat_result()

    def batch_u(self, batch_u):
        return np.asarray(batch_u, np.int32)

    def batch_seq(self, batch_seq):
        return np.asarray(batch_seq, np.int32)

    def batch_item_idx(self, batch_item_idx):
        return np.asarray(batch_item_idx, np.int32)

    def predict_top_k(self, sess: tf.Session, u, seq, item_idx, batch_size=0):
        data = [u, seq]
        batch_inputs = BatchInputAndConcatResult(data, batch_size=batch_size)

        item_idx = np.array(item_idx, dtype=np.int32)
        for batch_data in tqdm(batch_inputs, ncols=70, unit='b'):
            batch_u, batch_seq = batch_data
            batch_u = self.batch_u(batch_u)
            batch_seq = self.batch_seq(batch_seq)
            batch_item_idx = self.batch_item_idx(item_idx)
            feed_dict = {self.u: batch_u,
                         self.input_seq: batch_seq,
                         self.test_item: batch_item_idx,
                         self.is_training: False}
            batch_result = sess.run(self.test_top_k, feed_dict,
                options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
            batch_inputs.update_result(batch_result)
        return batch_inputs.concat_result()

    def sparse_ce_loss(self, pos, item_emb_table, item_bias_table, seq_emb, istarget):
        logits = tf.matmul(seq_emb, tf.transpose(item_emb_table))  # (N, M)
        if item_bias_table is not None:
            logits += tf.expand_dims(item_bias_table, axis=0)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos, logits=logits)
        loss = tf.reduce_sum(losses * istarget) / tf.reduce_sum(istarget)
        return loss