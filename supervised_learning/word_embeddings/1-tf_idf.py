#!/usr/bin/env python3
"""
Defines a function to create a TF-IDF embedding matrix.
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # Clean sentences: lowercase and strip punctuation
    cleaned_sentences = []
    for sentence in sentences:
        cleaned = re.sub(r'[^\w\s\']', '', sentence.lower())
        cleaned_sentences.append(cleaned)

    tokenized_sentences = [s.split() for s in cleaned_sentences]

    # Determine features
    if vocab is None:
        words = set()
        for sentence in tokenized_sentences:
            words.update(sentence)
        features = sorted(list(words))
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

    # Calculate Inverse Document Frequency (IDF)
    # IDF = log(s / df), where df is total documents containing term
    doc_freq = np.zeros(f)
    for j, feature in enumerate(features):
        for sentence in tokenized_sentences:
            if feature in sentence:
                doc_freq[j] += 1

    # Apply natural log IDF and normalize final TF-IDF embeddings per sentence
    idf = np.log(s / np.maximum(doc_freq, 1e-12))
    embeddings = tf * idf

    # L2 normalize each non-zero sentence embedding vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
