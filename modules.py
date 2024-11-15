from __future__ import print_function

import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def embedding_lookup(table, inputs: tf.Tensor):
    shape = inputs.get_shape().as_list()[1:] + table.get_shape().as_list()[-1:]
    sparse = tf.contrib.layers.dense_to_sparse(inputs, -1)
    output = tf.nn.embedding_lookup_sparse(table, sparse, None)
    output = tf.reshape(output, [-1, ] + shape)
    return output


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True,
              initializer='',
              initializer_stddev=0.02,
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if initializer == 'truncated_normal':
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.initializers.truncated_normal(stddev=initializer_stddev))
        else:
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           #initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        else:
            lookup_table = tf.concat((lookup_table, tf.zeros(shape=[1, num_units]),), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        # outputs = embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def positional_attention(batch_size, max_len, num_heads, positional_attention_initializer, scope, reuse):
    """

    Args:
        Q_: (h*N, T_q, C/h)
        K_: (h*N, T_k, C/h). Assume T_k = T_q
        num_heads:
        scope:
        reuse:

    Returns:
        Attention map of shape (h*N, T_q, T_q)
    """
    with tf.variable_scope(scope, reuse=reuse):
        if positional_attention_initializer == 'truncated_normal':
            initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif positional_attention_initializer == 'zeros':
            initializer = tf.zeros_initializer()
        else:
            raise ValueError()

        attention = tf.get_variable('positional_attention',
                                    dtype=tf.float32,
                                    shape=[num_heads, max_len, max_len],
                                    initializer=initializer)
        if num_heads > 1:
            attention = tf.split(attention, num_heads)
            attention = [tf.tile(a, [batch_size, 1, 1]) for a in attention]
            attention = tf.concat(attention, axis=0)
        else:
            attention = tf.tile(attention, [batch_size, 1, 1])
    return attention


def factorized_positional_attention(batch_size, max_len, k, num_heads, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        stddev = 0.01 ** 0.5 / k ** 0.25
        factor1 = tf.get_variable('factor1',
                                  dtype=tf.float32,
                                  shape=[num_heads, max_len, k],
                                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        factor2 = tf.get_variable('factor2',
                                  dtype=tf.float32,
                                  shape=[num_heads, max_len, k],
                                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        attention = tf.matmul(factor1, tf.transpose(factor2, perm=(0, 2, 1)))
        if num_heads > 1:
            attention = tf.split(attention, num_heads)
            attention = [tf.tile(a, [batch_size, 1, 1]) for a in attention]
            attention = tf.concat(attention, axis=0)
        else:
            attention = tf.tile(attention, [batch_size, 1, 1])
        return attention


def apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, attention, queries,
                    padding_value=-2 ** 32 + 1, attention_normalization='softmax', attention_temperature=1.0,
                    scope=None):
    with tf.name_scope(scope) as scope:
        # Scale
        attention = attention / (K_.get_shape().as_list()[-1] ** 0.5)
        # attention = tf.Print(attention, [attention[0], ], 'Attention after scale', summarize=100000)
        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        # key_masks = tf.Print(key_masks, [key_masks[0], ], 'key abs sum:', summarize=100000)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(attention) * padding_value
        # key_masks = tf.Print(key_masks, [key_masks[0], ], 'key_masks', summarize=100000)
        attention = tf.where(tf.equal(key_masks, 0), paddings, attention)  # (h*N, T_q, T_k)
        # attention = tf.Print(attention, [attention[0], ], 'Attention after key masking', summarize=100000)
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(attention[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(attention)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_value
            # masks = tf.Print(masks, [masks[0], ], 'masks', summarize=100000)
            attention = tf.where(tf.equal(masks, 0), paddings, attention)  # (h*N, T_q, T_k)
        # Activation
        if attention_normalization == 'softmax':
            if attention_temperature != 1.0:
                attention = attention / attention_temperature
                # print('attention_temperature: {}'.format(attention_temperature))
            attention = tf.nn.softmax(attention)  # (h*N, T_q, T_k)
        elif attention_normalization == 'sum':
            # attention = tf.Print(attention,
            #                      [attention[0], tf.reduce_sum(attention[0], axis=-1, keep_dims=True), ],
            #                      'Attention', summarize=100000)
            attention = attention / tf.reduce_sum(attention + 1e-6, axis=-1, keep_dims=True)
        else:
            raise ValueError
        # attention = tf.Print(attention, [attention[0], ], 'Attention after normalization[0]', summarize=100000)
        # attention = tf.Print(attention, [attention[0], ], 'Attention after normalization', summarize=100000)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        attention *= query_masks  # broadcasting. (N, T_q, C)

        tf.summary.image('{}/attention_activation'.format(scope), tf.expand_dims(attention, axis=-1))
        tf.summary.image('{}/attention_activation_row_normalized'.format(scope),
                         tf.expand_dims(attention / (tf.reduce_max(attention, keep_dims=True, axis=-1) + 1e-6), axis=-1))
        # Dropouts
        attention = tf.layers.dropout(attention, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum

        new_value = tf.matmul(attention, V_)  # ( h*N, T_q, C/h)
        return new_value


def multihead_attention(queries, 
                        keys,
                        num_units=None, 
                        num_heads=8,
                        attention_type='dot_product',
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        linear_projection_and_dropout=False,
                        args=None,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        batch_size = tf.shape(V)[0]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        new_values = 0
        if 'dot_product' in attention_type:
            print('Add dot product attention')
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, scope='apply_dot_product_attention')
            new_values += outputs
        if 'positional' in attention_type:
            print('Add positional attention')
            outputs = positional_attention(batch_size, args.maxlen, num_heads, args.positional_attention_initializer,
                                       scope='positional_attention', reuse=reuse)
            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, attention_temperature=args.attention_temperature,
                                      scope='apply_positional_attention')
            new_values += outputs
        if 'factorized_positional' in attention_type:
            print('Add factorized positional attention')
            outputs = factorized_positional_attention(
                batch_size, args.maxlen, args.factorized_positional_attention_k, num_heads,
                scope='factorized_positional_attention', reuse=reuse
            )
            outputs = apply_attention(K_, V_, causality, dropout_rate, is_training, keys, num_heads, outputs,
                                      queries, scope='apply_factorized_positional_attention')
            new_values += outputs

        outputs = tf.concat(tf.split(new_values, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        if linear_projection_and_dropout:
            outputs = tf.layers.dense(outputs, num_units, activation=None)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs


def get_activation_function(activation_name: str):
    activation_map = {
        'relu': tf.nn.relu,
        'gelu': lambda x: 0.5 * x * (1.0 + tf.math.erf(x / 1.4142135623730951))
    }
    return activation_map[activation_name.lower()]


def feedforward(inputs, 
                num_units=[2048, 512],
                inner_act='relu',
                scope="multihead_attention", 
                dropout_rate=0.2,
                inner_dropout=True,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": get_activation_function(inner_act), "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        if inner_dropout:
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs
