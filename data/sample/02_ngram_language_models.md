# Unit 2: N-gram Language Models

A **language model** assigns a probability to a sequence of words. Language models are used in
speech recognition, spelling correction, machine translation and text generation.

## The chain rule and the Markov assumption

By the chain rule of probability, the probability of a sentence w1 ... wn is

P(w1 ... wn) = P(w1) * P(w2 | w1) * P(w3 | w1 w2) * ... * P(wn | w1 ... wn-1)

Long histories are almost never seen in training data, so an **n-gram model** makes the
**Markov assumption**: the next word depends only on the previous n-1 words. A **bigram** model
(n = 2) uses P(wi | wi-1); a **trigram** model (n = 3) uses P(wi | wi-2 wi-1).

## Maximum likelihood estimation

The maximum likelihood estimate (MLE) of a bigram probability is a ratio of counts:

P(wi | wi-1) = C(wi-1 wi) / C(wi-1)

For example, if "machine" occurs 50 times in the corpus and "machine learning" occurs 20 times,
then P(learning | machine) = 20 / 50 = 0.4. Sentences are padded with start `<s>` and end
`</s>` symbols so that the first and last words also have a context.

## The sparsity problem

Any n-gram that never occurs in the training corpus gets probability zero under MLE, and a
single zero makes the probability of the whole sentence zero. Larger n makes this worse, because
the number of possible n-grams grows exponentially while the corpus stays the same size.

## Smoothing

Smoothing moves some probability mass from seen to unseen n-grams.

### Laplace (add-one) smoothing

Add one to every count before normalising. With vocabulary size V:

P_Laplace(wi | wi-1) = (C(wi-1 wi) + 1) / (C(wi-1) + V)

Laplace smoothing is simple but moves too much mass to unseen events when V is large, so it
performs poorly for language modelling. **Add-k smoothing** adds a fraction k < 1 instead of 1.

### Backoff and interpolation

**Backoff** uses the trigram estimate if the trigram was seen, otherwise falls back to the
bigram, then the unigram. **Linear interpolation** always mixes all orders:

P(wi | wi-2 wi-1) = l1 * P(wi | wi-2 wi-1) + l2 * P(wi | wi-1) + l3 * P(wi), with l1 + l2 + l3 = 1

The weights are tuned on a held-out set.

### Kneser-Ney smoothing

Kneser-Ney smoothing is the best-performing classical method. It subtracts a fixed discount from
each seen count and redistributes that mass using a **continuation probability**: how many
different contexts a word appears after, rather than how often it appears. "Francisco" is
frequent but almost always follows "San", so it gets a low continuation probability.

## Evaluating language models: perplexity

**Perplexity** is the inverse probability of a test set, normalised by the number of words N:

PP(W) = P(w1 ... wN) ^ (-1/N)

Lower perplexity means the model predicts the test data better. Perplexity can be read as the
weighted average number of choices the model is uncertain between at each step. Perplexities
are only comparable between models that use the same vocabulary.

## Limitations

N-gram models cannot capture dependencies longer than n-1 words, need a lot of memory for large
n, and treat "cat" and "dog" as completely unrelated symbols, so evidence about one never helps
with the other. Neural language models address these problems with word embeddings and
recurrent or transformer architectures.
