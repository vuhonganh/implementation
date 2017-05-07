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
    bSize = 64
    eSize = 128
    lr = 0.005
    n_train_steps = 40000
    n_sampled = 32
    reader = TreebankReader()
    vSize = reader.get_vocab_size()
    model = Word2Vec(vSize, eSize, n_sampled, lr)
    model.build_model()
    train(model, reader, n_train_steps, bSize, wSize)


def word_analogy(embeddings, ia, ib, ic):
    """
    a:b :: c:?  -> find 10 candidates d the analogy word of c given relation a:b
    example: man:woman :: king:queen
    :param embeddings: embedding matrix: vector representation of words 
    :param ia: index of word a
    :param ib: index of word b
    :param ic: index of word c 
    :return: 10 top indices of d = argmax_i ((xb - xa + xc)^T . xi) / ||xb - xa + xc|| 
    """
    # renormalize embeddings to make sure
    embeddings_norm = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    embeddings /= embeddings_norm
    xa = embeddings[ia]
    xb = embeddings[ib]
    xc = embeddings[ic]
    relation = xb - xa + xc
    prod = embeddings.dot(relation)
    prod_sorted_idx = np.argsort(prod)
    return prod_sorted_idx[-10:]


def closet_word(embeddings, idx):
    # renormalize embeddings to make sure
    embeddings_norm = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    embeddings /= embeddings_norm
    curvec = embeddings[idx]
    prod = embeddings.dot(curvec)
    prod_sorted_idx = np.argsort(prod)
    return prod_sorted_idx[-10:]


def test_closest_word():
    reader = TreebankReader()
    embeddings = np.load('finalEmbed.npy')
    tk = reader.get_tokens()
    id2Tk = reader.get_id2Tokens()
    idx = tk['film']
    idds = closet_word(embeddings, idx)
    closest_words = [id2Tk[idd] for idd in idds]
    print(closest_words)
    # the result is fair enough, but we need to filter out frequent words

if __name__ == '__main__':
    # main()
    test_closest_word()

