from __future__ import absolute_import
from __future__ import division

import string
import re
import pickle
import tensorflow as tf
import numpy as np
import random
import sys
import os.path
import math
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def load_wav_file(filename): #Taken from input_data.py, packaged with tensorflow
	with tf.Session(graph=tf.Graph()) as sess:
		wav_filename_placeholder = tf.placeholder(tf.string, [])
		wav_loader = io_ops.read_file(wav_filename_placeholder)
		wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
		
		return sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename}).audio.flatten()
		
def getMFCC(filename): #Modified from input_data.py, packaged with tensorflow
	with tf.Session(graph=tf.Graph()) as sess:
		wav_filename_placeholder = tf.placeholder(tf.string, [])
		wav_loader = io_ops.read_file(wav_filename_placeholder)
		wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
		spectrogram = contrib_audio.audio_spectrogram(
			wav_decoder.audio,
			window_size=550,
			stride=number_cells,
			magnitude_squared=True)
		mfcc = contrib_audio.mfcc(
			spectrogram, 44100, dct_coefficient_count=20)
		return sess.run(mfcc, feed_dict={wav_filename_placeholder: filename})

def save_wav_file(filename, wav_data, sample_rate):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })



text = pickle.load(open("Data Sources/arpatext.pkl", 'rb'))
random.seed()
train=[]
test=[]
validate=[]
trainlabels=[]
filename=[]
for val in range(1, 2200):
	value0=random.random()
	if value0 <= 0.1:
		test.append(val)
	elif value0 <= 0.9:
		train.append(val)
		trainlabels.append(text[val][-1])
		filename.append("Data Sources/Audio2-0/Audio2-"+str(val).zfill(2)+".wav")
	else:
		validate.append(val)



biggest=0

for number in range(0, len(train)):
	audiofile=load_wav_file("Data Sources/Audio2-0/Audio2-"+str(train[number]+1).zfill(2)+".wav")
	if len(audiofile)>biggest:
		biggest=len(audiofile)
for number in range(0, len(train)):
	audiofile=load_wav_file("Data Sources/Audio2-0/Audio2-"+str(train[number]+1).zfill(2)+".wav")
	if len(audiofile)<biggest:
		audiofile.resize(biggest)
	save_wav_file("Data Sources/Audio2-0/Audio2-"+str(train[number]).zfill(2)+".wav", audiofile, 44100)
length_in_ms=biggest/44100.*1000
number_cells=int((length_in_ms-550)/5)

tf.logging.set_verbosity(tf.logging.INFO)
mfccPlaceholder=tf.placeholder(tf.float32)

labelsMatrix=np.zeros((len(train), 6))
for val in range(0, len(train)):
	labelsMatrix[val][int(trainlabels[val])-1]=1

lstm=tf.contrib.rnn.BasicLSTMCell(281*20)
output, state=tf.nn.static_rnn(lstm, [tf.reshape(mfccPlaceholder, [len(filename), 281*20])], dtype=tf.float32)

labelsTensor=tf.placeholder(tf.float32)

weights = tf.Variable(tf.random_normal([281*20, 6]))
biases = tf.Variable(tf.random_normal([6]))
model = tf.matmul(output[-1], weights) + biases
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labelsTensor))
saver = tf.train.Saver([weights, biases])
optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)
check = tf.equal(tf.argmax(model, 1), tf.argmax(labelsTensor, 1))
correct = tf.reduce_sum(tf.cast(check, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    
	step = 0
	num_correct = 0.
	accuracy = 0.
	mfccarray=[]
	for i in range(0, len(train)):
		mfcc=getMFCC(filename[i])
		mfccarray.append(mfcc)
	while step < 20000:
		_, corr = sess.run([optimizer, correct],
			feed_dict={mfccPlaceholder: mfccarray, labelsTensor: labelsMatrix})
		num_correct += corr
		accuracy = 100*num_correct/(len(train))
		if step % 100 == 0 or step<100:
			print 'Step', step, '- Accuracy =', accuracy
		num_correct = 0
		step += 1
	saver.save(sess, os.getcwd()+"/output")