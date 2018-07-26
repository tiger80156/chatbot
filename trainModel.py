import tensorflow as tf
from seq2seqModel import model_input

# Setting the Hyperparameters
EPCHOS = 100
BATCH_SIZE = 64
RNN_SIZE = 512
NUM_LAYERS = 3
ENCODING_EMBEDING_SIZE = 512
DECODING_EMBEDING_SIZW = 512
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.9
MIN_LEARNING_RATE = 0.0001
KEEP_PROBABILITEY = 0.5

# Defining the seccsion
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model input
inputs, targets, lr, keep_prob = model_input()

# Setting the seq2seq model length
sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")


