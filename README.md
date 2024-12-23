# RWKV

RWKV-v4

RWKV-v5

RWKV-v6 (implemented)

Each model will have training code (linear/parallel scan) and inference code. For use tf.scan for less memory usage and tfp.math.scan_associative.


Save a model.weights.h5 with the train code and load the weights with the inference code. Make sure that both models are the the same architecture and replace any training RWKV blocks with the inference type.


