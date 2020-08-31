import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Embedding,Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

import numpy as np
import os

decoder_srt='srt'
decoder_end='end'
eng=[]
with open('europarl-v7.bg-en.en',mode='rt', encoding='utf-8') as f:
    eng1=f.readlines()
    for k in eng1:
        k=decoder_srt+ ' ' + k + ' ' + decoder_end
        eng.append(k)

with open('europarl-v7.bg-en.bg',mode='rt', encoding='utf-8') as f:
    bg=f.readlines()

total_words=10000

encoder_tokenizer=Tokenizer(num_words=total_words,oov_token="<OOV>")
encoder_tokenizer.fit_on_texts(bg)
encoder_words=encoder_tokenizer.word_index
encoder_seq= encoder_tokenizer.texts_to_sequences(bg)
print(encoder_seq[1])
encoder_max_len=max([len(i) for i in encoder_seq])
encoder_input_data=np.array(pad_sequences(encoder_seq,maxlen=encoder_max_len,padding='pre',truncating='post'),dtype='float')



dencoder_tokenizer=Tokenizer(num_words=total_words,oov_token="<OOV>")
dencoder_tokenizer.fit_on_texts(eng)
dencoder_words=dencoder_tokenizer.word_index
dencoder_seq= dencoder_tokenizer.texts_to_sequences(eng)
print(dencoder_seq[1])
dencoder_max_len=max([len(i) for i in dencoder_seq])
dencoder_pad=np.array(pad_sequences(dencoder_seq,maxlen=dencoder_max_len+1,padding='post',truncating='post'),dtype='float')
dencoder_input_data=dencoder_pad[:,:-1]
dencoder_output_data=dencoder_pad[:,1:]

dencoder_tokens=dict([(w,q) for q,w in dencoder_words.items()])

encoder_input=Input(shape=(None,))
encoder_embedding=Embedding(total_words, 128)(encoder_input)
encoder_gru1=GRU(128,return_sequences = True)(encoder_embedding)
encoder_gru2=GRU(128,return_sequences = True)(encoder_gru1)
encoder_gru3=GRU(128,return_sequences = False)(encoder_gru2)


# encoder_model=Sequential()
# encoder_model.add(Embedding(total_words, 128, input_length=encoder_max_len))
# encoder_model.add(Bidirectional(GRU(128,return_sequences = True)))
# encoder_model.add(Bidirectional(GRU(128,return_sequences = True)))
# encoder_outputs, state_h=encoder_model.add(Bidirectional(GRU(128,return_sequences = False,return_state=True)))
# encoder_outputs, state_h=encoder_model.output
dencoder_initial_state=Input(shape=(128,))
dencoder_input=Input(shape=(None,))
dencoder_embedding=Embedding(total_words, 128)(dencoder_input)
dencoder_gru1=GRU(128,return_sequences = True)(dencoder_embedding,initial_state=encoder_gru3)
dencoder_gru2=GRU(128,return_sequences = True)(dencoder_gru1,initial_state=encoder_gru3)
dencoder_gru3=GRU(128,return_sequences =True)(dencoder_gru2,initial_state=encoder_gru3)
dencoder_output=Dense(len(dencoder_tokens),activation='softmax')(dencoder_gru3)
# dencoder_model=Sequential()
# dencoder_model.add(Embedding(total_words, 128, input_length=dencoder_max_len))
# dencoder_model.add(Bidirectional(GRU(128,return_sequences = True)))(initial_state=[state_h])
# dencoder_model.add(Bidirectional(GRU(128,return_sequences = True)))(initial_state=encoder_model.output)
# dencoder_model.add(Bidirectional(GRU(128,return_sequences = True)))(initial_state=encoder_model.output)
# dencoder_model.add(Dense(len(dencoder_words),activation='softmax'))

model_train=Model(inputs=[encoder_input,dencoder_input],outputs=[dencoder_output])

model_encoder=Model(inputs=[encoder_input],outputs=[encoder_gru3])

#model_dencoder=Model(inputs=[dencoder_input,dencoder_initial_state],outputs=[dencoder_output])

model_train.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(lr=1e-3))

history=model_train.fit([encoder_input_data,dencoder_input_data],[dencoder_output_data],batch_size=38,epochs=2)

def predict(input_data):
    input_data=decoder_srt+ ' ' + input_data + ' ' + decoder_end
    input_data=encoder_tokenizer.texts_to_sequences([input_data])
    input_data=np.array(pad_sequences(input_data,maxlen=encoder_max_len,padding='pre',truncating='post'),dtype='float')
    
    initial_state=model_encoder.predict(input_data)
    count=0
    decoder_input=np.zeros((1,dencoder_max_len+1))
    decoder_input[0,count]=dencoder_words['srt']
    decoder_word=dencoder_words['srt']
    dencoded_output=''
    while((decoder_word!=dencoder_words['end']) and (count< dencoder_max_len)):
        count+=1
        x=model_train.predict([input_data,decoder_input])
        decoder_token=x[0,count,:]
        token_int = np.argmax(decoder_token)
        decoder_word=dencoder_tokens[token_int]
        dencoded_output+=' ' + decoder_word
        print(count)
    return dencoded_output

a=predict(bg[2])