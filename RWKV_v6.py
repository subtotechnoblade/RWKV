import tensorflow as tf
import numpy as np

# tf.config.run_functions_eagerly(True)
np.set_printoptions(threshold=np.inf)



global_dtype = tf.float32

class Unembedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.unembedding = tf.keras.layers.Dense(vocab_size, use_bias=True, dtype=dtype,
                                                 intitializer=tf.keras.initializers.random_uniform(minval=-5e-5,
                                                                                                           maxval=5e-5))
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, *args, **kwargs):
        logits = self.unembedding(inputs)
        probs = self.softmax(logits)
        return probs


class Mish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.softplus = tf.keras.layers.Activation(tf.keras.activations.softplus)
        self.tanh = tf.keras.layers.Activation(tf.keras.activations.tanh)

    def call(self, inputs, *args, **kwargs):
        return inputs * self.tanh(self.softplus(inputs))


class Squared_ReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        return self.relu(inputs) ** 2


class Multi_Headed_Dense(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_size, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = self.embed_size // self.num_heads
        # (b, c, h, e)
        # (b, c, h, e, e)

        self.denses = [tf.keras.layers.Dense(self.head_dim, use_bias=use_bias) for _ in range(self.num_heads)]
        # self.dense = tf.keras.layers.Dense(self.head_dim, name="grr")

    # @tf.function()
    def call(self, inputs):
        # return inputs
        # context_size = tf.shape(inputs)[1]
        # return tf.reshape(self.dense(inputs), (-1, context_size, self.num_heads, self.head_dim))
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        return tf.transpose(tf.stack([self.denses[i](inputs[i]) for i in range(self.num_heads)]), [1, 2, 0, 3])


class Time_Mix(tf.keras.layers.Layer):
    def __init__(self, layer_id, num_heads, embed_size, token_shift_hidden_dim=32, **kwargs):
        super().__init__(**kwargs)

        self.layer_id = layer_id
        self.embed_size = embed_size
        self.num_heads = num_heads
        if self.embed_size % self.num_heads != 0:
            raise ValueError("embed must be a multiple of heads")
        self.head_dim = self.embed_size // self.num_heads

        self.mu_x = self.add_weight(shape=(self.embed_size,),
                                  name=f"mu_x{self.layer_id}")

        self.key = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"key_{self.layer_id}")
        # self.key = tf.keras.layers.Dense(self.head_dim, name=f"Key_{self.layer_id}")
        # self.key_mu = self.add_weight(shape=(self.embed_size,),
        #                           name=f"key_mu_time{self.layer_id}")
        self.key_lambda = self.add_weight(shape=(self.embed_size,),
                                      name=f"key_lambda{self.layer_id}")
        self.key_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"key_A_matrix{self.layer_id}")
        self.key_B = tf.keras.layers.Dense(embed_size, name=f"key_B_matrix{self.layer_id}")


        self.value = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"value_{self.layer_id}")
        # self.value = tf.keras.layers.Dense(self.head_dim, name=f"Value_{self.layer_id}")
        # self.value_mu = self.add_weight(shape=(self.embed_size,),
        #                             name=f"mix_v_time{self.layer_id}")
        self.value_lambda = self.add_weight(shape=(self.embed_size,),
                                        name=f"value_lambda{self.layer_id}")
        self.value_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"value_A_matrix{self.layer_id}")
        self.value_B = tf.keras.layers.Dense(embed_size, name=f"value_B_matrix{self.layer_id}")


        self.receptance = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"receptance_{self.layer_id}")
        # self.receptance = tf.keras.layers.Dense(self.head_dim, name=f"Receptance_{self.layer_id}")
        # self.receptance_mu = self.add_weight(shape=(self.embed_size,),
        #                                  name=f"mix_r_time{self.layer_id}")
        self.receptance_lambda = self.add_weight(shape=(self.embed_size,),
                                             name=f"receptance_lambda{self.layer_id}")
        self.receptance_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"receptance_A_matrix{self.layer_id}")
        self.receptance_B = tf.keras.layers.Dense(embed_size, name=f"receptance_B_matrix{self.layer_id}")


        self.gate = tf.keras.layers.Dense(embed_size, use_bias=False, name=f"gate_{self.layer_id}")
        # self.gate_mu = self.add_weight(shape=(self.embed_size,),
        #                            name=f"mix_g_time{self.layer_id}")
        self.gate_lambda = self.add_weight(shape=(self.embed_size,),
                                        name=f"gate_lambda{self.layer_id}")
        self.gate_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"gate_A_matrix{self.layer_id}")
        self.gate_B = tf.keras.layers.Dense(embed_size, name=f"gate_B_matrix{self.layer_id}")

        self.decay = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"decay_{self.layer_id}")  # known in the formula as w
        # self.decay = tf.keras.layers.Dense(self.head_dim, name=f"Decay_{self.layer_id}")  # known in the formula as w
        # self.decay_mu = self.add_weight(shape=(self.embed_size,),
        #                             name=f"mix_w_time{self.layer_id}")
        self.decay_lambda = self.add_weight(shape=(self.embed_size,),
                                        name=f"decay_lambda{self.layer_id}")
        self.decay_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"decay_A_matrix{self.layer_id}")
        self.decay_B = tf.keras.layers.Dense(embed_size, name=f"decay_B_matrix{self.layer_id}")


        self.bonus = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"bonus_{self.layer_id}")
        # known in the formula as u
        # self.bonus = tf.keras.layers.Dense(self.head_dim, name=f"Bonus_{self.layer_id}")  # known in the formula as u
        # self.bonus_mu = self.add_weight(shape=(self.embed_size,),
        #                             name=f"mix_u_{self.layer_id}")
        self.bonus_lambda = self.add_weight(shape=(self.embed_size,),
                                        name=f"bonus_lambda{self.layer_id}")
        self.bonus_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"bonus_A_matrix{self.layer_id}")
        self.bonus_B = tf.keras.layers.Dense(embed_size, name=f"bonus_B_matrix{self.layer_id}")

        self.wkv_A_matrix = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"wkv_A_matrix{self.layer_id}")
        self.wkv_B_matrix = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"wkv_B_matrix{self.layer_id}")
        self.wkv_lamb = self.add_weight(shape=(self.num_heads, self.embed_size // self.num_heads,), name=f"wkv_lambda{self.layer_id}")

        self.gn = tf.keras.layers.GroupNormalization(self.num_heads, axis=-1, name=f"GroupNorm_{self.layer_id}")
        self.out = tf.keras.layers.Dense(self.embed_size, name=f"Out_{self.layer_id}")
    # self.built = True

    def lora(self, x, lamb, A_matrix, B_matrix):
        return lamb + B_matrix(tf.nn.tanh(A_matrix(x)))

    def lerp(self, a, b, mu):
        return a + (b - a) * mu
    def ddlerp(self, x, last_x, mu, lamb, A_matrix, B_matrix):
        diff = last_x - x
        return x + diff * self.lora(x + diff * mu, lamb, A_matrix, B_matrix)

    def token_shift_v6(self, x, last_x, mu, lamb, A_matrix, B_matrix):
        """
        :param x: current inputs
        :param last_x: token shifted inputs
        :param mu: trainable vector
        :param l: lambda trainable vector
        :param A: trainable matrix downsample in lora
        :param B: trainable matrix upsample in lora
        :return:

        # a is last_x, and b is current x or inputs
        # lora(x) = lambda + tanh(xA)B
        # ddlerp(a, b) = a + (b - a) * lora(a + (b - a) * mu_x) note that * means gated or element mul
        # output = x * mu + last_x * (1 - mu)
        # first compute (b - a) call this diff
        """
        batch_size, context_length, embed_size = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        output = self.ddlerp(x, last_x, mu, lamb, A_matrix, B_matrix)
         # split the output into individual heads
        return tf.reshape(output, [batch_size, context_length, self.num_heads, self.head_dim])
        # return tf.reshape(x, [batch_size, context_length, self.num_heads, self.head_dim])

    def token_shift(self, inputs):
        return tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1]

    def combine_fn(self, sum_a, prod_a, sum_b, prod_b):
        """
        Given arr = [(x0, 1), (x1, w0), (x2, w1)]
        :param a: accumulated result
        :param b: new elements from the initial tensor
        :return:
        """
        # print(sum_a.shape, prod_a.shape, sum_b.shape, prod_b.shape)
        # return sum_b + prod_a * sum_a, prod_b * prod_a
        return sum_b + prod_a * sum_a, prod_b

    # @tf.function(jit_compile=True)
    def serial_scam(self, u, w, kv):
        """
        Note that context_length
        :param u: bonus (batch, context_length, embed)
        :param w: decay (batch, context_length, embed)
        :param ktv: (batch, context_length, embed, embed) # the matrix update for every time step / token step
        :return: the matrix memory at every timestep according to https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v6.png
        """
        # decay (w) is [:, 1:] because in the equations w0 is simply not used and x0 goes with w1 not w0
        matrix_mem_update, product_acc = tfp.math.scan_associative(lambda a, b: self.combine_fn(*a, *b),
                                                                   (kv[:, :-1], tf.expand_dims(w[:, 1:], axis=-1)),
                                                                   axis=1)
        del product_acc

        matrix_mem = kv * tf.expand_dims(u, axis=-1)
        matrix_mem += tf.concat((tf.zeros_like(kv[:, :1], dtype=global_dtype), matrix_mem_update), axis=1)
        return matrix_mem

    # @tf.function()
    def serial_scam2(self, u, w, kv):
        # w = tf.concat((tf.ones_like(w[:, :1]), w[:, 1:]), axis=1)

        w = tf.expand_dims(w, axis=-1)
        w_reshaped = tf.transpose(w, [1, 0, 2, 3, 4])
        kv_reshaped = tf.transpose(kv, [1, 0, 2, 3, 4])
        # decay (w) is [1:] because in the equations w0 is simply not used and x0 goes with w1 not w0

        matrix_mem_update, _ = tf.scan(lambda a, b: self.combine_fn(*a, *b), (kv_reshaped[:-1], w_reshaped[1:]),
                                       parallel_iterations=32,
                                       swap_memory=True,
                                       infer_shape=False)
        matrix_mem_update = tf.transpose(matrix_mem_update, [1, 0, 2, 3, 4])
        matrix_mem = kv * tf.expand_dims(u, axis=-1)

        matrix_mem += tf.concat((tf.zeros_like(kv[:, :1]), matrix_mem_update), axis=1)
        return matrix_mem

    # @tf.function()
    def call(self, inputs, trainable=True, *args, **kwargs):
        batch_size, context_length, embed_size = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        x = inputs
        last_x = self.token_shift(inputs)

        mu_x = self.lerp(x, last_x, self.mu_x)

        k = self.key(self.token_shift_v6(x, last_x, mu_x, self.key_lambda, self.key_A, self.key_B))  # keys
        r = self.receptance(
            self.token_shift_v6(x, last_x, mu_x, self.receptance_lambda, self.receptance_A,
                                self.receptance_B))  # receptance


        w = self.decay(
            self.token_shift_v6(x, last_x, mu_x, self.decay_lambda, self.decay_A, self.decay_B))  # decay

        v = self.value(
            self.token_shift_v6(x, last_x, mu_x, self.value_lambda, self.value_A, self.value_B))  # value


        u = self.bonus(self.token_shift_v6(x, last_x, mu_x, self.bonus_lambda, self.bonus_A, self.bonus_B))


        g = self.gate(self.ddlerp(x, last_x, mu_x, self.gate_lambda, self.gate_A, self.gate_B))  # for gating we don't have heads

        # k and v: batch, h, context, embed, 1
        kv = tf.expand_dims(k, axis=-1) @ tf.expand_dims(v, axis=3)


        w = self.lora(w, tf.expand_dims(tf.expand_dims(self.wkv_lamb, 0), 0), self.wkv_A_matrix, self.wkv_B_matrix)
        w = tf.exp(-tf.exp(w))


        # wkv = self.serial_scam(u, w, kv)  # <- serial scan goes here
        # wkv1 = self.serial_scam1(u, w, ktv)  # <- serial scan goes here
        # wkv = self.serial_scam2(1, w, kv)  # <- serial scan goes here
        wkv = self.serial_scam2(u, w, kv)  # <- serial scan goes here
        # print(wkv.shape)
        rwkv = tf.squeeze(wkv @ tf.expand_dims(r, axis=-1), axis=-1)

        rwkv = tf.reshape(rwkv, [batch_size * context_length, embed_size])
        rwkv = self.gn(rwkv)
        rwkv = tf.reshape(rwkv, [batch_size, context_length, embed_size])

        rwkv *= tf.nn.silu(g)

        time_mixed = self.out(rwkv)
        return time_mixed


class Channel_Mix(tf.keras.layers.Layer):
    def __init__(self, layer_id, embed_size, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id  # not that this will always be odd
        self.embed_size = embed_size
        self.hidden_size = hidden_size if hidden_size is not None else int(self.embed_size * 3.5)

        self.key = tf.keras.layers.Dense(self.hidden_size)
        self.mu_k = self.add_weight(shape=(self.embed_size,), name=f"mu_k_channel_{self.layer_id}")

        self.value = tf.keras.layers.Dense(self.embed_size)

        self.receptance = tf.keras.layers.Dense(self.embed_size)
        self.mu_r = self.add_weight(shape=(self.embed_size,), name=f"mu_r_channel_{self.layer_id}")

        self.squared_ReLU = Squared_ReLU()
        # self.mish = Mish()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def token_shift(self, inputs):
        return tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1]

    def lerp(self, a, b, mu):
        return a + (b - a) * mu
    # @tf.function(jit_compile=True)
    def call(self, inputs, trainable=True, *args, **kwargs):
        x = inputs
        last_x = self.token_shift(inputs)

        k = self.key(self.lerp(x, last_x, self.mu_k))
        # k = self.mish(k)
        k = self.squared_ReLU(k)
        kv = self.value(k)

        r = self.receptance(self.lerp(x, last_x, self.mu_r))
        r = self.sigmoid(r)

        rkv = kv * r
        return rkv


class RWKV_Block(tf.keras.layers.Layer):
    def __init__(self, layer_id, num_heads, embed_size, token_shift_hidden_dim=32, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.time_mix = Time_Mix(layer_id, num_heads, embed_size, token_shift_hidden_dim, name=f"Time_Mix_{layer_id}")

        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.channel_mix = Channel_Mix(layer_id, embed_size, hidden_size)


    def call(self, inputs, trainable=True, *args, **kwargs):
        x = self.layer_norm1(inputs)
        x = self.time_mix(x)
        x += inputs
        residual = x
        x = self.layer_norm2(x)
        x = self.channel_mix(x)
        x += residual
        return x
def make_test_model(embed_size, num_heads=1, num_layers=1):
    inputs = tf.keras.layers.Input(batch_shape=(None, None, embed_size), name="inputs")
    x = inputs
    for layer_id in range(num_layers):
        x = RWKV_Block(layer_id, num_heads, embed_size)(x)


    return  tf.keras.Model(inputs=inputs, outputs=x)
if __name__ == "__main__":
    import numpy as np

    model = make_test_model(64, 4, 1)
    model.summary()

    x_train = np.random.randint(low=0, high=100, size=(4, 2, 64))
    print(x_train.shape)
    y_train = 2 * x_train + 1

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-2),
                  loss="mse",
                  metrics=["acc", "mae"])
    model.fit(x_train, y_train, epochs=150)
    x = np.random.randint(low=0, high=100, size=(4, 2, 64))
    print(x[0][0])
    print()
    print(model(x)[0][0])



