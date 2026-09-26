#!/usr/bin/env python3
"""
Defines a function to create a bag-of-words embedding matrix.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    tokenized_sentences = []
    vocab_set = set()

    for sentence in sentences:
        # Lowercase, replace non-alphanumeric characters (except apostrophes) with spaces
        cleaned = ''.join(
            c if c.isalnum() or c == "'" else ' ' for c in sentence
        ).lower()
        words = cleaned.split()
        
        # Keep tokens and maintain apostrophes properly
        tokenized_sentences.append(words)
        if vocab is None:
            vocab_set.update(words)

    if vocab is None:
        features = sorted(list(vocab_set))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    feature_index = {word: i for i, word in enumerate(features)}

    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, features
