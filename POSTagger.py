import random
from collections import Counter
import nltk
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_addresses_file(file_path):
    """
    Read 'tab' delimited SOTU (State of the Union) data file
    :param file_path: Path of file to read
    :return: List of addresses
    """
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path,
                         sep='\t',
                         header=None,
                         usecols=range(3),
                         names=["year", "president", "address text"])
        return df
    else:
        print('File does not exist in the specified path')


def pos_tag(s):
    """
    Tag the given text to its corresponding Part Of Speech (POS) tags.
    Use the NLTK package.
    :param s: String sentence to tag.
    :return: List of POS tags.
    """
    tokenized = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(tokenized)
    return tagged


def recognize_ne(s):
    """
    Recognize named entities in given sentence.
    Use the NLTK package.
    :param s: String sentence to tag.
    :return: Tree structure of NE recognition.
    """
    ne_tree = nltk.ne_chunk(s, binary=False)
    iob_tags = nltk.tree2conlltags(ne_tree)
    return iob_tags


def count_unigram_tags(tagging, normalize=False):
    """
    Count the number of occurrences of each unigram POS tag in a given tagging.
    :param tagging: POS tagging of sentence.
    :param normalize: Binary flag indicating whether counts are normalized.
    :return: Dictionary structure of counts.
    """
    length = len(tagging)
    observed_tags = [t[1] for t in tagging]
    counts = Counter(observed_tags)
    if normalize:
        normalized = {}
        for key in counts:
            normalized[key] = float(counts[key]) / length
        return normalized
    return counts


def count_bigrams_tags(tagging, normalize=False):
    """
    Count the number of occurrences of each bigram POS tag in a given tagging.
    :param tagging: POS tagging of sentence.
    :param normalize: Binary flag indicating whether counts are normalized.
    :return: Dictionary structure of counts.
    """
    length = len(tagging)
    bigrams = list(nltk.bigrams(tagging))
    bigram_pos = ((pos1, pos2) for (w1, pos1), (w2, pos2) in bigrams)
    bigram_counts = Counter(bigram_pos)
    if normalize:
        normalized = {}
        for key in bigram_counts:
            normalized[key] = float(bigram_counts[key]) / length
        return normalized
    return bigram_counts


def count_trigrams_tags(tagging, normalize=False):
    """
    Count the number of occurrences of each trigram POS tag in a given tagging.
    :param tagging: POS tagging of sentence.
    :param normalize: Binary flag indicating whether counts are normalized.
    :return: Dictionary structure of counts.
    """
    length = len(tagging)
    trigrams = list(nltk.trigrams(tagging))
    trigram_pos = ((pos1, pos2, pos3) for (w1, pos1), (w2, pos2), (w3, pos3) in trigrams)
    trigram_counts = Counter(trigram_pos)
    if normalize:
        normalized = {}
        for key in trigram_counts:
            normalized[key] = float(trigram_counts[key]) / length
        return normalized
    return trigram_counts


def main():
    # nltk.download()
    df = read_addresses_file(r"C:\Users\micha\Downloads\Res.tsv")

    pos_unigrams_NNP = []
    pos_unigrams_PRP = []
    pos_unigrams_DT = []
    total_pos_bigrams_count = Counter()
    total_pos_trigrams_count = Counter()
    for index, row in df.iterrows():
        tagging = pos_tag(row['address text'])
        pos_unigrams_counts = count_unigram_tags(tagging, normalize=True)
        pos_unigrams_NNP.append(pos_unigrams_counts['NNP'])
        pos_unigrams_PRP.append(pos_unigrams_counts['PRP'])
        pos_unigrams_DT.append(pos_unigrams_counts['DT'])

        total_pos_bigrams_count += count_bigrams_tags(tagging)
        total_pos_trigrams_count += count_trigrams_tags(tagging)

    df['NNP_unigrams_count'] = pos_unigrams_NNP
    df['PRP_unigrams_count'] = pos_unigrams_PRP
    df['DT_unigrams_count'] = pos_unigrams_DT

    df.plot(x='year', y=['NNP_unigrams_count', 'PRP_unigrams_count', 'DT_unigrams_count'])
    plt.ylabel("Normalized POS Count")
    plt.xticks(np.arange(1790, 2017, 25))
    plt.show()

    ###############
    """
    Find the most frequent bigrams and trigrams in corpus
    """
    most_common_bi = total_pos_bigrams_count.most_common(5)
    most_common_tri = total_pos_trigrams_count.most_common(5)
    print("5 most frequent bigrams in the corpus are {0}".format(most_common_bi))
    print("5 most frequent trigrams in the corpus are {0}".format(most_common_tri))
    ###############

    """
    Find most frequent unigrams, bigrams and trigrams in the speeches of the presidents:
    George Washington, Abraham Lincoln, Richard Nixon, Ronald Reagan, Barack Obama, Donald J. Trump
    """
    presidents = ['George Washington',
                  'Abraham Lincoln',
                  'Richard Nixon',
                  'Ronald Reagan',
                  'Barack Obama',
                  'Donald J. Trum']

    for president in presidents:
        sliced_df = df.loc[df['president'] == president]

        total_pos_unigrams_count = Counter()
        total_pos_bigrams_count = Counter()
        total_pos_trigrams_count = Counter()

        for index, row in sliced_df.iterrows():
            tagging = pos_tag(row['address text'])
            total_pos_unigrams_count += count_unigram_tags(tagging, normalize=False)
            total_pos_bigrams_count += count_bigrams_tags(tagging, normalize=False)
            total_pos_trigrams_count += count_trigrams_tags(tagging, normalize=False)

        total = sum(total_pos_unigrams_count.values(), 0.0)
        for key in total_pos_unigrams_count:
            total_pos_unigrams_count[key] /= total
        most_common_uni = total_pos_unigrams_count.most_common(5)

        for key in total_pos_bigrams_count:
            total_pos_bigrams_count[key] /= total
        most_common_bi = total_pos_bigrams_count.most_common(5)

        for key in total_pos_trigrams_count:
            total_pos_trigrams_count[key] /= total
        most_common_tri = total_pos_trigrams_count.most_common(5)

        print("5 most frequent unigrams of president {0} are {1}".format(president, most_common_uni))
        print("5 most frequent bigrams of president {0} are {1}".format(president, most_common_bi))
        print("5 most frequent trigrams of president {0} are {1}".format(president, most_common_tri))

    #############
    """
    Sample 10 sentences of the speeches of Washington and Clinton, for further analysis.
    """
    presidents = ['George Washington',
                  'William J. Clinto']

    for president in presidents:
        sliced_df = df.loc[df['president'] == president]
        speeches = sliced_df['address text']
        sampled_speech = random.sample(list(speeches), 1)

        sentences = nltk.sent_tokenize(sampled_speech[0])
        sampled_sentences = random.sample(sentences, 10)
        for i, sentence in enumerate(sampled_sentences):
            print("Sentence {0} of president {1}: {2}".format(i, president, pos_tag(sentence)))


if __name__ == '__main__':
    main()
