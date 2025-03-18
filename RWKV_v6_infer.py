import tensorflow as tf
# import tensorflow_probability as tfp
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

    # @tf.function()
    def call(self, inputs):
        # return inputs
        # Note that the inputs shape is (batch, heads, head_dim)
        # thus we slice on the first axis
        inputs = tf.transpose(inputs, [1, 0 ,2])
        x = tf.transpose(tf.squeeze(tf.stack([self.denses[i](inputs[i: i + 1] ) for i in range(self.num_heads)]), axis=1), [1, 0, 2])
        return x

class Time_Mix(tf.keras.layers.Layer):
    def __init__(self, layer_id, num_heads, embed_size, token_shift_hidden_dim=32, **kwargs):
        super().__init__(**kwargs)

        self.layer_id = layer_id
        self.embed_size = embed_size
        self.num_heads = num_heads
        if self.embed_size % self.num_heads != 0:
            raise ValueError("embed must be a multiple of heads")
        self.head_dim = self.embed_size // self.num_heads

        self.mu_x = self.add_weight(shape=(self.embed_size,), name=f"mu_x{self.layer_id}")

        self.key = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"key_{self.layer_id}")
        # self.key = tf.keras.layers.Dense(self.head_dim, name=f"Key_{self.layer_id}")
        # self.key_mu = self.add_weight(shape=(self.embed_size,),
        #                           name=f"key_mu{self.layer_id}")
        self.key_lambda = self.add_weight(shape=(self.embed_size,),
                                      name=f"key_lambda{self.layer_id}")
        self.key_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"key_A_matrix{self.layer_id}")
        self.key_B = tf.keras.layers.Dense(embed_size, name=f"key_B_matrix{self.layer_id}")


        self.value = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"value_{self.layer_id}")
        # self.value = tf.keras.layers.Dense(self.head_dim, name=f"Value_{self.layer_id}")
        # self.value_mu = self.add_weight(shape=(self.embed_size,),
        #                             name=f"value_mu_time{self.layer_id}")
        self.value_lambda = self.add_weight(shape=(self.embed_size,), name=f"value_lambda{self.layer_id}")
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


        self.bonus = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"bonus_{self.layer_id}")  # known in the formula as u
        # self.bonus = tf.keras.layers.Dense(self.head_dim, name=f"Bonus_{self.layer_id}")  # known in the formula as u
        # self.bonus_mu = self.add_weight(shape=(self.embed_size,),
        #                             name=f"mix_u_{self.layer_id}")
        self.bonus_lambda = self.add_weight(shape=(self.embed_size,), name=f"bonus_lambda{self.layer_id}")
        self.bonus_A = tf.keras.layers.Dense(token_shift_hidden_dim, name=f"bonus_A_matrix{self.layer_id}")
        self.bonus_B = tf.keras.layers.Dense(embed_size, name=f"bonus_B_matrix{self.layer_id}")

        self.wkv_A_matrix = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"wkv_A_matrix{self.layer_id}")
        self.wkv_B_matrix = Multi_Headed_Dense(num_heads, embed_size, use_bias=False, name=f"wkv_B_matrix{self.layer_id}")
        self.wkv_lamb = self.add_weight(shape=(self.num_heads, self.embed_size // self.num_heads,), name=f"wkv_lambda{self.layer_id}")

        self.gn = tf.keras.layers.GroupNormalization(self.num_heads, axis=-1, name=f"GroupNorm_{self.layer_id}")
        self.out = tf.keras.layers.Dense(self.embed_size, name=f"Out_{self.layer_id}")
        # self.built = True

    def update_tensor(self, input_tensor, indexes, update, depth=1):
        #  returns a new tensor which can be used to overwrite the input_tensor
        if depth == 1:
            axis_0 = indexes[0]
            output_tensor = tf.concat((input_tensor[:axis_0], update, input_tensor[axis_0:]), axis=0)
            return output_tensor
        elif depth == 2:
            axis_0, axis_1 = indexes
            # print(input_tensor[axis_0][:axis_1].shape, update.shape, input_tensor[axis_0][axis_1 + 1:].shape)
            axis_0_update = tf.concat((input_tensor[axis_0][:axis_1], update, input_tensor[axis_0][axis_1 + 1:]),
                                      axis=0)
            axis_0_update = tf.expand_dims(axis_0_update, axis=0)

            output_tensor = tf.concat((input_tensor[:axis_0], axis_0_update, input_tensor[axis_0 + 1:]), axis=0)
            return output_tensor

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
        batch_size = tf.shape(x)[0]
        output = self.ddlerp(x, last_x, mu, lamb, A_matrix, B_matrix)
        # split the output into individual heads
        return tf.reshape(output, [batch_size, self.num_heads, self.head_dim])




    # @tf.function()
    def call(self, inputs, state, state_matrix, trainable=True, *args, **kwargs):
        x = inputs
        last_x = state[self.layer_id][0]

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

        kv = tf.expand_dims(k, axis=-1) @ tf.expand_dims(v, axis=2)

        w = self.lora(w, tf.expand_dims(self.wkv_lamb, 0), self.wkv_A_matrix, self.wkv_B_matrix)
        w = tf.expand_dims(tf.exp(-tf.exp(w)), axis=-1)
        u = tf.expand_dims(u, axis=-1)


        wkv = kv * u + state_matrix[self.layer_id]

        state_matrix = tf.tensor_scatter_nd_update(state_matrix, tf.constant([[self.layer_id]]),
                                                   tf.expand_dims(w * state_matrix[self.layer_id] + kv, axis=0))

        rwkv = tf.squeeze(wkv @ tf.expand_dims(r, axis=-1), axis=-1)

        rwkv = tf.reshape(rwkv, [-1, self.embed_size])
        rwkv = self.gn(rwkv)

        state = tf.tensor_scatter_nd_update(state, [[self.layer_id, 0]], tf.expand_dims(inputs, 0))

        # state = self.update_tensor(state, [self.layer_id, 0], inputs, depth=1)

        time_mixed = self.out(rwkv * tf.nn.silu(g))
        return time_mixed, state, state_matrix


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

    def lerp(self, a, b, mu):
        return a + (b - a) * mu


    # @tf.function(jit_compile=True)
    def call(self, inputs, state, trainable=True, *args, **kwargs):
        x = inputs
        last_x = state[self.layer_id][1]

        k = self.key(self.lerp(x, last_x, self.mu_k))
        # k = self.mish(k)
        k = self.squared_ReLU(k)
        kv = self.value(k)

        r = self.receptance(self.lerp(x, last_x, self.mu_r))
        r = self.sigmoid(r)
        rkv = kv * r

        state = tf.tensor_scatter_nd_update(state, [[self.layer_id, 1]], tf.expand_dims(inputs, 0))
        return rkv, state


class RWKV_Block(tf.keras.layers.Layer):
    def __init__(self, layer_id, num_heads, embed_size, token_shift_hidden_dim=32, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.time_mix = Time_Mix(layer_id, num_heads, embed_size, token_shift_hidden_dim, name=f"Time_Mix_{layer_id}")

        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.channel_mix = Channel_Mix(layer_id, embed_size, hidden_size)


    def call(self, inputs, state, state_matrix, trainable=True, *args, **kwargs):
        x = self.layer_norm1(inputs)
        x, state, state_matrix = self.time_mix(x, state, state_matrix)
        x += inputs
        residual = x
        x = self.layer_norm2(x)
        x, state = self.channel_mix(x, state)
        x += residual
        return x, state, state_matrix

if __name__ == "__main__":
    import numpy as np
    from RWKV_v6 import make_test_model
    embed_size, num_heads, num_layers = 64, 4, 5


    model = make_test_model(embed_size, num_heads, num_layers)
    model.summary()

    # model(np.random.randint(low=0, high=100, size=(1, 2, embed_size)))

    model.save_weights("test_model.weights.h5")


    def make_test_model_infer(embed_size, num_heads=1, num_layers=1):
        inputs = [tf.keras.layers.Input(batch_shape=(None, embed_size), name="inputs"),
        tf.keras.layers.Input(batch_shape=(num_layers, 2, embed_size), name="input_state"),
        tf.keras.layers.Input(batch_shape= (num_layers, num_heads, embed_size // num_heads, embed_size // num_heads), name="inputs_state_matrix")]

        x, state, state_matrix = inputs

        for layer_id in range(num_layers):
            x, state, state_matrix = RWKV_Block(layer_id, num_heads, embed_size)(x, state, state_matrix)
        output_state, output_state_matrix = state, state_matrix

        return tf.keras.Model(inputs=inputs, outputs=[x, output_state, output_state_matrix])

    model_infer = make_test_model_infer(embed_size, num_heads, num_layers)
    # model_infer.summary()

    def create_states():
        return np.zeros((num_layers, 2, embed_size)), np.zeros(
            (num_layers, num_heads, embed_size // num_heads, embed_size // num_heads))
    input_state, input_state_matrix = create_states()


    model_infer.load_weights("test_model.weights.h5")

    dummy_data = np.random.uniform(low=0, high=100, size=(1, 360, embed_size))

    output1 = model(dummy_data)
    # print(output1)

    output2 = []
    for data_point in dummy_data[0]:
        data_point = np.expand_dims(data_point, 0)

        x, input_state, input_state_matrix = model_infer([data_point,input_state, input_state_matrix])
        input_state = input_state.numpy()
        input_state_matrix = input_state_matrix.numpy()
        output2.append(x)

    output1 = np.array(output1).reshape(-1, 64)
    output2 = np.array(output2).reshape(-1, 64)
    print(output1.shape, output2.shape)

    index = 1
    print(np.allclose(output1, output2))
    for index in range(len(output1)):
        # print(output1[index], output2[index])
        print(np.sum(np.abs(output1[index] - output2[index])))




