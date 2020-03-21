import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataOperator as do

import SkipgramNetwork as sgn

def main():
    train_data_path = sys.argv[1]
    output_weight_path = sys.argv[2]
    output_tokenizer_path = sys.argv[3]
    
    tokenizer, word_count, train_target, train_context, train_label = do.load_train_data(train_data_path)

    skipgram_model = sgn.create_model(word_count)
    skipgram_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    skipgram_model.summary()

    skipgram_model.fit([train_target, train_context], train_label, epochs=cfg.epochs)

    do.save_embed_weight(skipgram_model.get_weights()[0][1:], output_weight_path)
    do.save_text_tokenizer(tokenizer, output_tokenizer_path)

if __name__ == '__main__':
    main()