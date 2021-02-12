from flask import Flask, render_template, request
from google.protobuf import message
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation,InputLayer,Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy,categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import IPython
from keras.utils import to_categorical
import sys

print(sys.version)

app = Flask(__name__,template_folder='template')
df=pd.read_csv("dataset.csv")

class Vocab_builder():
  '''
  Builds vocabulary and 
  word to index and index to word dictionaries
  from dataset
  '''
  def __init__(self,lang,series):
    self.lang=lang
    self.data=series
  def tokenize(self,line):
    return line.split(' ')
  def build_vocab(self):
    self.uniq_words=set()
    
    self.maxlen=0
    count=3
    self.num_list=[]
    for index,line in self.data.items():
      self.word_list=self.tokenize(line)
      self.maxlen=max(len(self.word_list),self.maxlen)
      for word in self.word_list:
        if(word not in self.uniq_words and word!='<EOS>' and word!='<SOS>'):
          self.uniq_words.add(word)
          self.num_list.append(count)
          count+=1
      
    self.vocab_list=['<PAD>','<SOS>','<EOS>']+sorted(list(self.uniq_words))
    self.num_list=[0,1,2]+self.num_list
    print("Built vocabulary having {} elements".format(len(self.vocab_list)))
    print("Largest sentence length (with tags):{}".format(self.maxlen))
    return dict(zip(self.vocab_list,self.num_list)),dict(zip(self.num_list,self.vocab_list))

eng=Vocab_builder('eng',df['eng'])
ger=Vocab_builder('ger',df['ger'])
eng_w2i,eng_i2w=eng.build_vocab()
ger_w2i,ger_i2w=ger.build_vocab()

def process_text(lines):
    lines=lines.strip()
    lines=''.join(c for c in unicodedata.normalize('NFD', lines) if unicodedata.category(c) != 'Mn')
    lines=lines.encode('utf8','ignore').decode('utf8')
    lines=lines.replace(u'\u200b',' ')
    lines=lines.lower()
    lines=lines.replace('\xa0', ' ')
    lines=re.sub(r"([.,?!;])",r" \1 ",lines)   #adding spaces before and after punctuation
    lines=re.sub(r"[0-9]"," ",lines)
    lines=re.sub(r'["]'," ",lines)
    lines=re.sub(r"[']","",lines)
    lines=re.sub(r"[%-,]"," ",lines)
    lines=re.sub(r"[:]"," ",lines)
    lines=re.sub(r'[" "]+'," ",lines)  #removing excess spaces
    lines=lines.strip() #removing spaces from the end of string
    lines="<SOS> "+lines+" <EOS>"
    return lines

def sent_to_ind(sentence,lang):
  '''
  Tokenizes a string and
  converts it to an np array of 
  indices and pads the 
  array according to max sentence length
  '''
  ind_list=[]
  if lang=='eng':
    tokens=eng.tokenize(sentence)
    for token in tokens:
      ind_list.append(eng_w2i[token])
    while len(ind_list)<max(ger.maxlen,eng.maxlen):
      ind_list.append(0)
  else:
    tokens=ger.tokenize(sentence)
    for token in tokens:
      ind_list.append(ger_w2i[token])
    while len(ind_list)<max(ger.maxlen,eng.maxlen):
      ind_list.append(0)
    
  return np.array(ind_list)


def sent_to_np(series,lang,translate_mode):
  '''
  Converts a dataframe column to 
  a unsqueezed np array of indexes
  with padding for feeding into NN
  '''
  ret_list=[]
  if translate_mode==False :
    if lang=='eng':
      for index,val in series.items():
        ret_list.append(sent_to_ind(val,'eng'))
    else:
      for index,val in series.items():
        ret_list.append(sent_to_ind(val,'ger'))
    
    ret_list=np.array(ret_list)
    return np.expand_dims(ret_list,axis=2)
  else:
    ans=sent_to_ind(series,'eng')
    ans=np.expand_dims(ans,axis=0)
    ans=np.expand_dims(ans,axis=2)
    return ans

def translate(sentence,mod,embedded):
  '''
  Function for translating given English sentence
  to German using model predictions
  '''
  ans=""
  preproc_sent=process_text(sentence)
  #print(preproc_sent)
  model_inp=sent_to_np(preproc_sent,'eng',True)
  if (embedded):
    model_inp=np.squeeze(model_inp,axis=2)
  #print(model_inp)
  pred=mod.predict(model_inp)
  #print(pred)
  for i in pred[0]:
    ind=np.argmax(i)
    #print(ind)
    #print("MAX:",i[ind])
    #print(i[24])
    if(ger_i2w[ind]=='<SOS>' or ger_i2w[ind]=='<EOS>' or ger_i2w[ind]=='<PAD>'):
        continue
    ans+=ger_i2w[ind]
    ans+=" "
  return ans

model=tf.keras.models.load_model("embedded_20000_final.h5")



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
		    return render_template('index.html')
    if request.method == 'POST':
      sen = request.form.get("eng_input") 
      try:
        res_sen=translate(sen,model,True)
      except KeyError as e:
        return render_template('result.html', prediction="Out of vocabulary word",message=e)
      
      return render_template('result.html', prediction=res_sen,message="Translated successfully")



if __name__ == '__main__':
	app.run(debug=True)