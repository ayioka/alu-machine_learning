#!/usr/bin/env python3
"""
Defines a function to create a bag-of-words embedding matrix.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # Clean sentences: lowercase and strip punctuation (preserve apostrophes)
    cleaned_sentences = []
    for sentence in sentences:
        cleaned = re.sub(r'[^\w\s\']', '', sentence.lower())
        cleaned_sentences.append(cleaned)

    # Tokenize words per sentence
    tokenized_sentences = [s.split() for s in cleaned_sentences]

    # Determine vocabulary / features list if not provided
    if vocab is None:
        words = set()
        for sentence in tokenized_sentences:
            words.update(sentence)
        features = sorted(list(words))
    else:
        features = list(vocab)

    # Map features to indices
    feature_index = {word: i for i, word in enumerate(features)}

    # Build embedding matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, features
