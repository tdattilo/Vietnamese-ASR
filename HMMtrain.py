"""
Written by Thomas Dattilo, April 2018 for Machine Learning 767 final project
This program takes text collected from various sources and eventually
trains a bigram-based hidden Markov model intended for homophone disambiguation for ASR. 
The process for this is somewhat simplified by the fact that the "hidden"
states are observable in text for training -- they are just not observable
from audio, which is why the hidden Markov model becomes necessary. 

Because of this simplification, I just train a normal Markov chain and then
give each node in the graph a corresponding custom ARPAbet-analogue observable
value. The observable states list for each node is sparse - one entry of probability 1
and the rest with probability 0.
"""
from __future__ import division
import string
import pickle
import math
import re
import numpy as np
def findTone(word):
	secondTone=re.compile("[ìềèừầờằàùồòỳ]")
	if secondTone.search(word)!=None:
		return '2'
	thirdTone=re.compile("[ỉểẻửẩởẳảủổỏỷ]")
	if thirdTone.search(word)!=None:
		return '3'
	fourthTone=re.compile("[ĩễẽữẫỡẵãũỗõỹ]")
	if fourthTone.search(word)!=None:
		return '4'
	fifthTone=re.compile("[íếéứấớắáúốóý]")
	if fifthTone.search(word)!=None:
		return '5'
	sixthTone=re.compile("[ịệẹựậợặạụộọỵ]")
	if sixthTone.search(word)!=None:
		return '6'
	else:
		return '1';


def alphaToArpa(word):
	tonemark=findTone(word)
	newword=word

	dictionary={ 'ì':'i',
				 'ỉ':'i',
				 'ĩ':'i',
				 'í':'i',
				 'ị':'i',
				 'ề':'#',
				 'ể':'#',
				 'ễ':'#',
				 'ế':'#',
				 'ệ':'#',
				 'ê':'#',
				 'è':'e',
				 'ẻ':'e',
				 'ẽ':'e',
				 'é':'e',
				 'ẹ':'e',
				 'ừ':'{',
				 'ử':'{',
				 'ữ':'{',
				 'ứ':'{',
				 'ự':'{',
				 'ư':'{',
				 'ầ':'@',
				 'ẩ':'@',
				 'ẫ':'@',
				 'ấ':'@',
				 'ậ':'@',
				 'â':'@',
				 'ờ':'}',
				 'ở':'}',
				 'ỡ':'}',
				 'ớ':'}',
				 'ợ':'}',
				 'ơ':'}',
				 'ằ':'!',
				 'ẳ':'!',
				 'ẵ':'!',
				 'ắ':'!',
				 'ặ':'!',
				 'ă':'!',
				 'à':'a',
				 'ả':'a',
				 'ã':'a',
				 'á':'a',
				 'ạ':'a',
				 'ù':'u',
				 'ủ':'u',
				 'ũ':'u',
				 'ú':'u',
				 'ụ':'u',
				 'ồ':'$',
				 'ổ':'$',
				 'ỗ':'$',
				 'ố':'$',
				 'ộ':'$',
				 'ô':'$',
				 'ò':'o',
				 'ỏ':'o',
				 'õ':'o',
				 'ó':'o',
				 'ọ':'o',
				 'ỳ':'y',
				 'ỷ':'y',
				 'ỹ':'y',
				 'ý':'y',
				 'ỵ':'y',
				 'c':'k',
				 'q':'k',
				 'd':'z',
				 'r':'z',
				 'đ':'d'}
	newword=newword.translate(newword.maketrans(dictionary))
	newword=re.sub(r'i#u','j@u',newword)
	newword=re.sub(r'i[a#]','i@',newword)
	newword=re.sub(r'oai', 'wai', newword)
	newword=re.sub(r'u$i','w@i', newword)
	newword=re.sub(r'uy#','wi@', newword)
	newword=re.sub(r'{}i', '{}j', newword)
	newword=re.sub(r'ngh', '~', newword)
	newword=re.sub(r'ng', '~', newword)
	newword=re.sub(r'tz', 'r', newword)
	newword=re.sub(r'gi.+', 'z', newword)
	newword=re.sub(r'gi$', 'zi', newword)
	newword=re.sub(r'gh', 'g', newword)
	newword=re.sub(r'ch', 'c', newword)
	newword=re.sub(r's', '%', newword)
	newword=re.sub(r'x', '%', newword)
	newword=re.sub(r'kh', 'x', newword)
	newword=re.sub(r'anh', 'ainh', newword)
	newword=re.sub(r'nh', '(', newword)
	newword=re.sub(r'ph', 'f', newword)
	newword=re.sub(r'th', ')', newword)
	newword=re.sub(r'ay','ey',newword)
	newword=re.sub(r'@y','#y',newword)
	newword=re.sub(r'ao','au',newword)
	newword=re.sub(r'@o','@u',newword)
	newword=re.sub(r'eo','eu',newword)
	newword=re.sub(r'oa', 'wa', newword)
	newword=re.sub(r'o!', 'w!', newword)
	newword=re.sub(r'oe', 'we', newword)
	newword=re.sub(r'ua', 'u@', newword)
	newword=re.sub(r'{a', '{@', newword)
	newword=re.sub(r'u$', 'u@', newword)
	newword=re.sub(r'ac', 'aik', newword)
	newword=re.sub(r'!m', 'aum', newword)
	newword=re.sub(r'!~', 'au~', newword)
	newword=re.sub(r'$m', '$um', newword)
	newword=re.sub(r'$~', '$u~', newword)
	return newword+tonemark;

class HMM:
	def __init__(self):
		self.allwords=[]
		self.wordvalues={}
		self.train()
	def train(self):
		f=open("file.txt", "r")
		trainingtext=f.read()
		f.close()
		trainingtext="<s> " + trainingtext
		trainingtext=trainingtext.translate(str.maketrans({a:" </s> <s>" for a in [".", "?", "!"]}))
		trainingtext=trainingtext.translate(str.maketrans({a:None for a in ["(", ")", "[", "]", ",", "'", '"']}))
		trainingtext=trainingtext.split()
		for word in trainingtext:
			if not(word.lower() in self.allwords):
				self.allwords.append(word.lower())
				self.allwords=sorted(self.allwords)

		for word in self.allwords:
			if not (word=="<s>") and not (word=="</s>"):
				self.wordvalues.update({word:alphaToArpa(word)})
		self.wordvalues.update({"<s>":""})
		self.wordvalues.update({"</s>":""})
		self.size=len(self.allwords)
		runningtotal=0.0
		self.bigrammodel=np.zeros((self.size, self.size))
		for number in range(1, len(trainingtext)):
			if not (trainingtext[number].lower() == "<s>"):
				self.bigrammodel[self.allwords.index(trainingtext[number].lower())][self.allwords.index(trainingtext[number+1].lower())]+=1
				runningtotal+=1.0
		self.bigrammodel=self.bigrammodel/runningtotal

	def getViterbi(self, soundstring):
		viterbi=np.zeros((self.size, len(soundstring)))
		backpointer=np.zeros((self.size, len(soundstring)))
		obsmultiplier=0.
		for number in range(1, self.size):
			if soundstring[0]==self.wordvalues[self.allwords[number]]:
				obsmultiplier = 1.
			viterbi[number][0]=self.bigrammodel[0][number]*obsmultiplier
		for time in range(1, len(soundstring)-1):
			print(time)
			obsmultiplier=0.
			for state in range(0, self.size):
				if soundstring[time]==self.wordvalues[self.allwords[state]]:
					obsmultiplier = 1.
				viterbi[state][time]=max(viterbi[earlystate][time-1]*self.bigrammodel[earlystate][state]*obsmultiplier for earlystate in range(0, self.size))
				argmax=0
				maxholder=0
				for earlystate in range(0, self.size):
					if maxholder<viterbi[earlystate][time-1]*self.bigrammodel[earlystate][state]:
						argmax=earlystate
						maxholder=viterbi[earlystate][time-1]*self.bigrammodel[earlystate][state]
				backpointer[state][time]=argmax
		viterbi[self.size-1][len(soundstring)-1]=max(viterbi[state][len(soundstring)-1]*self.bigrammodel[state][len(soundstring)-1] for state in range(0, self.size))
		argmax=0
		maxholder=0
		for earlystate in range(0, self.size):
			if maxholder<viterbi[earlystate][len(soundstring)-1]*self.bigrammodel[earlystate][self.size-2]:
				argmax=earlystate
				maxholder=viterbi[earlystate][len(soundstring)-1]*self.bigrammodel[earlystate][self.size-2]
		print(argmax)
		backpointer[self.size-1][len(soundstring)-1]=argmax
		state=self.size-1
		"""
		for word in range(len(soundstring)-1, 0, -1):
			translatestring = backpointer[
		"""		
		translatestring=""
		return translatestring

hmmtrained = HMM()
testValue="Vở kịch do đạo"
testValue=testValue.split()
for word in range(0, len(testValue)):
	testValue[word]=alphaToArpa(testValue[word].lower())
print(hmmtrained.getViterbi(testValue))