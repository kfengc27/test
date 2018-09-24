import tensorflow as tf
import re 
import string
import numpy as np

BATCH_SIZE = 150

MAX_WORDS_IN_REVIEW = 300  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_LAYERS = 2

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})



def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    #Remove stop words 
    review = review.lower() 

    review_words = review.split(' ')

    for word in stop_words:
        if word in review_words:
            review_words.remove(word)

    #Remove some necessary punctutaion in the sentence 
    for p in string.punctuation:
        if p in review_words:
            review_words.remove(p)

    #paraphrasing
    processed_review = review_words
    return processed_review

#Reference: https://github.com/tensorflow/tensorflow/issues/16186
def lstm_cell():
   lstm = tf.contrib.rnn.BasicLSTMCell(BATCH_SIZE, forget_bias=1.0, state_is_tuple=True, dtype=tf.float32)
   lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=tf.placeholder_with_default(0.75, shape=[], name='dropout_keep_prob')) 
   return lstm

#Version 2 
# def define_graph():

#     input_data= tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
#     labels = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,2], name='labels')
#     dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')

#     lstm = tf.nn.rnn_cell.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True, dtype=tf.float32)
#     lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
#     # l_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)]) #Double lstm
    
#     init = lstm.zero_state(BATCH_SIZE, dtype=tf.float32)
#     outputs, states= tf.nn.dynamic_rnn(lstm, input_data, initial_state=init)
#     # weight = tf.Variable(tf.truncated_normal([BATCH_SIZE, 2]))
#     # bias = tf.Variable(tf.constant(0.1, shape=[2]))
#     # results = tf.layers.dense(states[1], 2, activation=tf.nn.sigmoid)
#     # results = tf.layers.dense(states[1], 1, tf.nn.relu)
#     # results = tf.matmul(states[1], weight) + bias
#     # batch_range = tf.range(tf.shape(outputs)[0])
#     # indices = tf.stack([batch_range, BATCH_SIZE-1], axis=1)
#     # res = tf.gather_nd(outputs, indices)
#     # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
#     logits= tf.layers.dense(inputs=states.h, units=2, activation=None)

#     loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels), name='loss')
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

#     correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

#     return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

#VERSION 3.0
def define_graph():

    input_data= tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
    labels = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,2], name='labels')
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')

    lstm = tf.nn.rnn_cell.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True, dtype=tf.float32)
    lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
    
    init = lstm.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs, states= tf.nn.dynamic_rnn(lstm, input_data, initial_state=init)
    # weight = tf.Variable(tf.truncated_normal([BATCH_SIZE, 2]))
    # bias = tf.Variable(tf.constant(0.1, shape=[2]))
    # results = tf.layers.dense(states[1], 2, activation=tf.nn.sigmoid)
    # results = tf.layers.dense(states[1], 1, tf.nn.relu)
    # results = tf.matmul(states[1], weight) + bias
    # batch_range = tf.range(tf.shape(outputs)[0])
    # indices = tf.stack([batch_range, BATCH_SIZE-1], axis=1)
    # res = tf.gather_nd(outputs, indices)
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    logits= tf.layers.dense(inputs=states.h, units=2, activation=None)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels), name='loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

# VERSION 1.0  
# def define_graph():
#     input_data= tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
#     labels = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,2], name='labels')
#     dropout_keep_prob = tf.placeholder_with_default(0.75, shape=[], name='dropout_keep_prob')

#     # lstm = tf.nn.rnn_cell.BasicLSTMCell(BATCH_SIZE, forget_bias=1.0, state_is_tuple=True, dtype=tf.float32)
#     # lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)

#     l_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)]) #Double lstm
#     # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])
#     init = l_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
#     drop_first = tf.contrib.rnn.DropoutWrapper(l_cell, output_keep_prob=dropout_keep_prob)
#     outputs, init = tf.nn.dynamic_rnn(drop_first, input_data, dtype=tf.float32)
#     # dense = tf.layers.dense(outputs[:,-1], 50, activation=tf.nn.relu)
#     # drop_second = tf.layers.dropout(dense, rate=(1-dropout_keep_prob))
#     # results = tf.layers.dense(drop_second, 2, activation=None)
#     weight = tf.Variable(tf.truncated_normal([BATCH_SIZE, 2]))

#     bias = tf.Variable(tf.constant(0.1, shape=[2]))
#     outputs = tf.transpose(outputs, [1, 0, 2])
#     results = tf.matmul(init[2], weight) + bias
#     loss=tf.reduce_max(tf.nn.softmax_cross_entropy_with_logits(logits=results,labels=labels),name="loss")
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#     # correct_prediction = tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1))
#     preds = tf.nn.softmax(results)
#     correct_prediction = tf.equal(tf.argmax(preds,1),tf.argmax(labels,1))
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")
#     return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

# # def lstm_cell():
# #     return tf.contrib.rnn.BasicLSTMCell(150)

# # def define_graph():
# #     """
# #     Implement your model here. You will need to define placeholders, for the input and labels,
# #     Note that the input is not strings of words, but the strings after the embedding lookup
# #     has been applied (i.e. arrays of floats).
# #     In all cases this code will be called by an unaltered runner.py. You should read this
# #     file and ensure your code here is compatible.
# #     Consult the assignment specification for details of which parts of the TF API are
# #     permitted for use in this function.
# #     You must return, in the following order, the placeholders/tensors for;
# #     RETURNS: input, labels, optimizer, accuracy and loss
# #     """
