# Unit 3: Word Embeddings

A **word embedding** represents a word as a dense vector of real numbers, typically with 50 to
300 dimensions, such that words used in similar contexts get similar vectors.

## From one-hot vectors to dense vectors

A **one-hot vector** has one dimension per vocabulary word, with a 1 for the word and 0
everywhere else. One-hot vectors are huge and sparse, and every pair of words is equally
dissimilar: the dot product of any two different one-hot vectors is 0.

The **distributional hypothesis** ("you shall know a word by the company it keeps", Firth 1957)
says that words appearing in similar contexts have similar meanings. Dense embeddings put this
idea into practice.

## Count-based vectors: TF-IDF and co-occurrence

Before neural embeddings, words and documents were represented with counts. **TF-IDF** weights a
term t in document d by its term frequency multiplied by its inverse document frequency:

tf-idf(t, d) = tf(t, d) * log(N / df(t))

where N is the number of documents and df(t) is the number of documents containing t. Rare terms
get a higher weight. TF-IDF vectors are sparse and long; dimensionality reduction with SVD
(latent semantic analysis) turns co-occurrence counts into dense vectors.

## Word2Vec

Word2Vec (Mikolov et al., 2013) learns embeddings with a shallow neural network trained on a
simple prediction task over a sliding context window.

### Skip-gram

The **skip-gram** model predicts the surrounding context words from the centre word. With a
window of size 2, the centre word "learning" in "deep learning models need data" is trained to
predict "deep", "models" and "need". Skip-gram works well for small datasets and represents rare
words well.

### CBOW

The **continuous bag-of-words (CBOW)** model does the opposite: it predicts the centre word from
the average of its context word vectors. CBOW trains several times faster than skip-gram and is
slightly better for frequent words, but it smooths over rare words.

### Negative sampling

Computing a softmax over the whole vocabulary at every step is expensive. **Negative sampling**
replaces it with a binary classification task: tell the true context word apart from k randomly
sampled "negative" words (typically k = 5 to 20). Negative words are sampled in proportion to
their unigram frequency raised to the power 3/4.

## GloVe and fastText

**GloVe** (Global Vectors) fits word vectors so that the dot product of two word vectors
approximates the logarithm of how often the words co-occur, combining global count statistics
with the efficiency of prediction-based training.

**fastText** represents each word as the sum of vectors of its character n-grams (for example
"<wh", "whe", "her", "ere", "re>" for "where"). It can therefore build vectors for words never
seen in training, which helps with typos and morphologically rich languages.

## Properties and evaluation

Similarity between embeddings is measured with **cosine similarity**:
cos(u, v) = (u . v) / (|u| |v|), which ranges from -1 to 1.

Embeddings capture analogies through vector arithmetic: vector("king") - vector("man") +
vector("woman") is closest to vector("queen"). Intrinsic evaluation uses word-similarity and
analogy benchmarks; extrinsic evaluation measures how much the embeddings improve a downstream
task such as named entity recognition.

## Limitations

Static embeddings give each word exactly one vector, so the two senses of "bank" (river bank and
financial bank) share a single representation. They also absorb social biases present in the
training text. **Contextual embeddings** from models such as ELMo and BERT solve the first
problem by computing a different vector for each occurrence of a word based on its sentence.
