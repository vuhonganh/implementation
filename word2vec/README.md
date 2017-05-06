In Word2Vec model, we slide a window through the text corpus, and at each time we
* either try to predict the center word given the surrounding words in that window (CBOW)
* or predict surrounding words given the center word (Skip-gram).
Note that the word2vec model is based on these conditioned probabilities to achieve a vector representation of word that minimize the penalty of wrong predictions. Hence, in each window, the given words are also called input word, and the others are called output word. 

# Skip-gram model
Denote $U$, $V$ the output and input embedding matrix respectively. In a window, denote center word is $c$ and one output word is $o$. The prediction of $o$ is made using softmax function:
$$
\hat{y_o} = p(o | c) = \frac{exp(u_o^T v_c)}{\sum_{w=1}^W exp(u_w^T v_c)} 
$$