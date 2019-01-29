import nltk
import numpy as np
import tensorflow as tf
import gensim, logging, os
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def Sentence2vec(file):
	values=[]
	max_sentences = 15
	max_words = 70
	vector_representation_size = 100

	with open(file) as f:
		for line in f:
			list1=nltk.word_tokenize(line.strip())
			values.append(list1)
	model = Word2Vec(values,size = vector_representation_size, window = 6,min_count=1,workers = 4)
	fname = 'prabha_output.model'
	#print(model['సిపిఎం'])
	vector = np.zeros(shape=(max_sentences,max_words,vector_representation_size))
	for i,value in enumerate(values):
		for j,word in enumerate(value):
			vector[i][j] = model[word]
	vector = tf.cast(vector, tf.float32)

	# sentence to Vector code starts here....
	pooled_outputs = []

	sequence_length = max_words
	embedding_size = vector_representation_size
	filter_sizes = [3,4,5,6]
	num_filters = 3
	embedded_chars_expanded = tf.expand_dims(vector, -1)

	for i,filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):

			# Convolution Layer
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") # weights
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")		 # Biases
			conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv") #VALID" padding means that we slide the filter over our sentence without padding the edges,
			# Apply nonlinearity
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

			# Max-pooling over the outputs
			pooled = tf.nn.max_pool(
					    h,
					    ksize=[1, sequence_length - filter_size + 1, 1, 1],
					    strides=[1, 1, 1, 1],
					    padding='VALID',
					    name="pool")
			pooled_outputs.append(pooled)
	print(len(pooled_outputs))

	# Combine all the pooled features
	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(pooled_outputs,3)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(h_pool_flat))

Sentence2vec('sentences.txt')