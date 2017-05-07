from data.process_data import *
from models.word2vec import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train(model, data_reader, n_train_steps, bSize, wSize):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log', sess.graph)
        sess.run(tf.global_variables_initializer())
        avg_loss = 0.0
        for i in range(n_train_steps):
            batch_input, batch_label = data_reader.gen_batch(bSize, wSize)
            _, loss_batch, summary = sess.run([model.train_op, model.loss, model.summary_op],
                                              {model.input_placeholder: batch_input,
                                               model.label_placeholder: batch_label})
            writer.add_summary(summary, global_step=i)
            avg_loss += loss_batch
            if (i + 1) % 1000 == 0:
                print('step {}: {:5.4f}'.format(i + 1, avg_loss/1000))
                avg_loss = 0.0
        final_embed = sess.run(model.embeddings)
        np.save('finalEmbed.npy', final_embed)


def main():
    wSize = 4
    bSize = 16
    eSize = 128
    lr = 0.005
    n_train_steps = 40000
    n_sampled = 32
    reader = TreebankReader()
    vSize = reader.get_vocab_size()
    model = Word2Vec(vSize, eSize, n_sampled, lr)
    model.build_model()
    train(model, reader, n_train_steps, bSize, wSize)

if __name__ == '__main__':
    main()

