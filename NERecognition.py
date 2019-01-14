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
    s_t = pos_tag(s)
    ne_tree = nltk.ne_chunk(s_t, binary=False)
    # iob_tags = nltk.tree2conlltags(ne_tree)
    entities = []
    entities.extend(extract_entity_names(ne_tree))
    return entities


def count_ne(tagging):
    """
    Count the number of occurrences of each named entity in a given tagging.
    :param tagging: Given tagging of sentence.
    :return: Dictionary structure of counts.
    """
    length = len(tagging)
    if tagging:
        ne_counts = Counter(tagging)
        normalized = {}
        for key in ne_counts:
            normalized[key] = float(ne_counts[key]) / length
        return normalized
    else:
        return None


def extract_entity_names(ne_tree):
    """
    Traverse the NE tree structure to extract the named entities.
    :param ne_tree: Tree structure of NE's.
    :return: List of the NEs and their tag.
    """
    ne_in_sent = []
    for subtree in ne_tree:
        if type(subtree) == nltk.Tree:  # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))
    return ne_in_sent


def main():
    # nltk.download()
    df = read_addresses_file(r".\Corpus\Res.tsv")

    ne_PERSON_count = []
    ne_GPE_count = []
    ne_ORG_count = []
    total_GPE_ne_count = Counter()
    all_GPE_ne = []

    """
    Extract the NEs with tag PERSON or GPE or ORGANIZATION and count their number of occurrences.
    """
    for index, row in df.iterrows():
        ne_tagging = recognize_ne(row['address text'])
        observed_tags = [t[1] for t in ne_tagging]
        all_GPE_ne = [t[0] for t in ne_tagging if t[1] == 'GPE']
        ne_counts = count_ne(observed_tags)

        if 'PERSON' in ne_counts:
            ne_PERSON_count.append(ne_counts['PERSON'])
        else:
            ne_PERSON_count.append(0)
        if 'GPE' in ne_counts:
            ne_GPE_count.append(ne_counts['GPE'])
        else:
            ne_GPE_count.append(0)
        if 'ORGANIZATION' in ne_counts:
            ne_ORG_count.append(ne_counts['ORGANIZATION'])
        else:
            ne_ORG_count.append(0)

    df['PERSON_count'] = ne_PERSON_count
    df['GPE_count'] = ne_GPE_count
    df['ORG_count'] = ne_ORG_count

    df.plot(x='year', y=['PERSON_count', 'GPE_count', 'ORG_count'])
    plt.ylabel("Normalized NE Count")
    plt.xticks(np.arange(1790, 2017, 25))
    plt.show()

    ########################
    """
    Find most frequent NEs tagged as GPE.
    """
    total_GPE_ne_count = Counter(all_GPE_ne)

    most_common_GPE = total_GPE_ne_count.most_common(5)
    print("5 most common GPE words: {0}".format(most_common_GPE))

    year = []
    count = []
    for index, row in df.iterrows():
        year.append(row['year'])
        count.append(row['address text'].count(most_common_GPE[0][0]))
    year_GPE_count = {'year': year, 'count': count}
    df_GPE = pd.DataFrame(year_GPE_count)
    # year_GPE_count[row['year']] = row['address text'].count(most_common_GPE[0][0])
    # df_GPE = pd.DataFrame.from_dict(year_GPE_count, orient='index')
    df_GPE.plot(x='year', y='count', kind='line')
    plt.ylabel("Count of 'American' NE")
    plt.xticks(np.arange(1790, 2017, 25))
    plt.show()

    ########################
    ########################
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
            print("Sentence {0} of president {1}: {2}".format(i, president, recognize_ne(sentence)))


if __name__ == '__main__':
    main()
