# Unit 1: Text Preprocessing and Tokenization

These notes cover the first step of almost every NLP pipeline: turning raw text into a sequence
of units that a model can count, look up or embed.

## What is a token?

A **token** is a unit of text that a system treats as one symbol. Depending on the method, a
token can be a word ("language"), a punctuation mark ("."), a subword piece ("##ization") or a
single character. **Tokenization** is the process of splitting a string into tokens. The set of
all distinct tokens a system knows is its **vocabulary**, and tokens not in the vocabulary are
called **out-of-vocabulary (OOV)** tokens.

Types and tokens are different counts: in "the cat saw the dog" there are 5 tokens but only 4
types, because "the" occurs twice. Heaps' law describes how the vocabulary grows with corpus
size: V = k * N^beta, where N is the number of tokens and beta is usually between 0.4 and 0.6.

## Word-level tokenization

The simplest approach splits on whitespace. This fails on punctuation ("end." becomes one token)
and on languages such as Chinese or Japanese that do not put spaces between words. Rule-based
tokenizers add regular expressions for punctuation, contractions ("don't" -> "do" + "n't"),
numbers, URLs and emoticons. The Penn Treebank tokenizer is a well-known rule-based example.

Word-level vocabularies have two problems: they become very large, and any word not seen during
training becomes an unknown token, usually written [UNK].

## Subword tokenization

Subword methods keep frequent words whole and split rare words into smaller, reusable pieces, so
there are no true OOV words and the vocabulary stays a fixed size (typically 30k to 50k).

### Byte Pair Encoding (BPE)

BPE starts from a vocabulary of single characters and repeatedly merges the most frequent
adjacent pair of symbols in the training corpus into a new symbol. For example, if "e" + "r" is
the most frequent pair, every "e r" becomes "er". Merging stops when the vocabulary reaches the
target size. At test time the learned merges are applied in the same order. GPT models use a
byte-level variant of BPE.

### WordPiece

WordPiece, used by BERT, is similar to BPE but chooses the merge that most increases the
likelihood of the training data rather than the most frequent pair. Pieces that continue a word
are marked with "##", so "tokenization" may become "token" + "##ization".

### Unigram language model

SentencePiece's unigram model starts with a large candidate vocabulary and removes the pieces
whose removal hurts the corpus likelihood least. It treats whitespace as an ordinary symbol
(written as "▁"), so it works directly on raw text in any language.

## Normalisation

Before or after tokenization, text is usually normalised:

- **Case folding**: lowercasing so that "Apple" and "apple" match. It can lose information
  (the company Apple versus the fruit).
- **Unicode normalisation**: NFKC turns ligatures such as "ﬁ" into "fi" and full-width
  characters into standard ones.
- **Stopword removal**: dropping very frequent function words such as "the", "of" and "is".
  Useful for keyword search, harmful for tasks that depend on word order or negation.

## Stemming versus lemmatization

**Stemming** chops word endings with heuristic rules, so "studies", "studying" and "studied" may
all become "studi". The Porter stemmer is the classic algorithm. Stemming is fast but can produce
non-words and can conflate unrelated words.

**Lemmatization** maps each word to its dictionary form (lemma) using a vocabulary and
morphological analysis, often with the part of speech: "better" -> "good", "was" -> "be",
"studies" -> "study". It is slower and needs language resources, but its output is always a
real word.

Rule of thumb: stemming is good enough for search engines and bag-of-words models; lemmatization
is preferred when the output is shown to people or when meaning matters.
