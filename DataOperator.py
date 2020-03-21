import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def load_text_tokenizer(tokenizer_path):
  with open(tokenizer_path, 'rb') as tokenizer_file:
      return pickle.load(tokenizer_file)

def save_text_tokenizer(tokenizer, tokenizer_path):
  with open(tokenizer_path, 'wb') as tokenizer_file:
      pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_embed_weight(weight_path):
  with open(weight_path, 'rb') as weight_file:
      return pickle.load(weight_file)

def save_embed_weight(weight, weight_path):
  with open(weight_path, 'wb') as weight_file:
      pickle.dump(weight, weight_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_train_data(train_data_path):
    train_data_file = open(train_data_path, 'r')
    train_data_text = train_data_file.read()
    train_data_file.close()
    train_data_text = train_data_text.lower().replace('\r', '').replace('\n', ' ')

    tokenizer = kr.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([train_data_text])
    encoded_train_data_text = tokenizer.texts_to_sequences([train_data_text])[0]

    word_count = len(tokenizer.word_index) + 1
    sampling_table = kr.preprocessing.sequence.make_sampling_table(word_count)
    train_couple, train_label = kr.preprocessing.sequence.skipgrams(encoded_train_data_text, vocabulary_size=word_count, window_size=cfg.window_size, sampling_table=sampling_table)

    train_target, train_context = zip(*train_couple)
    train_target = np.array(train_target)
    train_context = np.array(train_context)

    return tokenizer, word_count, train_target, train_context, train_label