#!/usr/bin/env python3
"""
Defines a function to create a TF-IDF embedding matrix.
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for the analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    tokenized_sentences = []
    vocab_set = set()

    for sentence in sentences:
        words = ''.join(
            c if c.isalnum() or c in "'-" else ' ' for c in sentence
        ).lower().split()
        cleaned_words = [w.strip("'") for w in words if w.strip("'")]
        tokenized_sentences.append(cleaned_words)
        if vocab is None:
            vocab_set.update(cleaned_words)

    if vocab is None:
        features = sorted(list(vocab_set))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    tf = np.zeros((s, f))

    # Calculate Term Frequency (TF)
    for i, sentence in enumerate(tokenized_sentences):
        words_in_sentence = len(sentence)
        if words_in_sentence == 0:
            continue
        for word in sentence:
            if word in features:
                j = features.index(word)
                tf[i, j] += 1
        tf[i] = tf[i] / words_in_sentence

    # Calculate Inverse Document Frequency (IDF) based on all documents
    doc_freq = np.zeros(f)
    for j, feature in enumerate(features):
        for sentence in tokenized_sentences:
            if feature in sentence:
                doc_freq[j] += 1

    # IDF = log(s / df)
    idf = np.log(s / np.maximum(doc_freq, 1))
    embeddings = tf * idf

    # L2 normalize each sentence embedding vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
