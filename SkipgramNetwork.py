import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def create_model(word_count, pre_train_file=None):

    input_target = kr.layers.Input(shape=(1,), name='input_target')
    input_context = kr.layers.Input(shape=(1,), name='input_context')
    
    embed = kr.layers.Embedding(input_dim=word_count, output_dim=cfg.embed_size, input_length=1, name='embed1')

    embed_target = embed(input_target)
    embed_target = kr.layers.Reshape(target_shape=(cfg.embed_size,), name='reshape_target')(embed_target)
    embed_context = embed(input_context)
    embed_context = kr.layers.Reshape(target_shape=(cfg.embed_size,), name='reshape_context')(embed_context)

    dot_product = kr.layers.Dot(axes=1, name='dot1')([embed_target, embed_context])
    dot_product = kr.layers.Reshape(target_shape=(1,), name='reshape_dot')(dot_product)

    output = kr.layers.Dense(units=1, activation='sigmoid', name='fc1')(dot_product)

    model = kr.models.Model(inputs=[input_target, input_context], outputs=output)

    if pre_train_file is not None:
        model.load_weights(pre_train_file, by_name=True)

    return model