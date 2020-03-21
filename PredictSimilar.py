import sys

import tensorflow as tf
import tensorflow.keras as kr
import sklearn.metrics as skm

import DataOperator as do

def predict_similar_word(tokenizer, index_to_word, distance_matrix, input_word, predict_count):
    input_word = input_word.lower()

    encoded = tokenizer.texts_to_sequences([input_word])[0][0]
    sorted_similar_word_list = distance_matrix[encoded - 1].argsort()[1:]

    word_list = list()
    for index in sorted_similar_word_list[0:predict_count]:
        word_list.append(index_to_word[index + 1])

    return word_list

def main():
    input_weight_path = sys.argv[1]
    input_tokenizer_path = sys.argv[2]
    input_word = sys.argv[3]
    input_predict_count = int(sys.argv[4])
    
    weight = do.load_embed_weight(input_weight_path)
    tokenizer = do.load_text_tokenizer(input_tokenizer_path)

    index_to_word = {}
    for word, index in tokenizer.word_index.items():
        index_to_word[index] = word

    distance_matrix = skm.pairwise.euclidean_distances(weight)

    print(predict_similar_word(tokenizer, index_to_word, distance_matrix, input_word, input_predict_count))

if __name__ == '__main__':
    main()