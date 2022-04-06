import os
import sys
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import math

stemming = SnowballStemmer(language='english')

# returns paths of all documents
def getDocumentsInCorpus(corpus_path, list_them_bool):
    documentsList = []
    try:
        for path, dirtry, files in os.walk(corpus_path):
            for iterator in files:
                documentsList.append(path + "\\" + iterator)
                if list_them_bool is True:
                    print(path + "\\" + iterator)
        return documentsList
    except:
        return None


# initializes dictionaries related to the targeted document
def initializeDictionaries(document_target, number_of_documents):
    word_occurrences_per_document = dict()  # counts total word occurrences, key is word, value is integer
    word_score = dict()  # serves as a word score holder
    sentence_dict = dict()  # serves as a sentence score holder
    try:
        with open(document_target, "r", encoding='utf-8') as file_target:
            text_target = file_target.read()
            sentences_target = sent_tokenize(text_target)
            for line_target in sentences_target:
                sentence_dict[line_target] = 0  # key is sentence, value is determined by words (currently 0)
                words_in_line = word_tokenize(line_target)
                for word in words_in_line:
                    if word.isalnum():
                        neo = stemming.stem(word)
                        exists = word_occurrences_per_document.get(neo, -1)
                        if exists == -1:  # if word doesn't exist already in dictionaries
                            word_occurrences_per_document[neo] = [0 for i_fun in range(number_of_documents)].copy()
                            word_score[neo] = 0
        return word_occurrences_per_document, word_score, sentence_dict
    except:
        return None, None, None


# counts word occurences of existing words in dictionary that counts them
def countWordOccurencesInCorpus(documents, word_occurrences_per_document):
    for i in range(len(documents)):
        with open(documents[i], "r", encoding='utf-8') as file:
            text = file.read()
            sentences = sent_tokenize(text)
            for line in sentences:
                line = word_tokenize(line)
                for word in line:
                    if word.isalnum():
                        neo = stemming.stem(word)
                        if neo in word_occurrences_per_document.keys():
                            word_occurrences_per_document[neo][i] = word_occurrences_per_document[neo][i] + 1
    return word_occurrences_per_document


# for each word in dictionary uses tf-idf formula to calculate its score
def calculateWordsScores(num_of_documents, index_of_target_document, word_occurrences_per_document, word_score):
    for key in word_occurrences_per_document:
        occurrences_in_target_document = word_occurrences_per_document[key][index_of_target_document]
        cnt = 0
        for i in range(num_of_documents):
            k = word_occurrences_per_document[key][i]
            if k:
                cnt += 1
        word_score[key] = occurrences_in_target_document * math.log(num_of_documents / cnt)
    return word_score


# print 10 (or less if word count is less than 10) words with highest scores
def printWordsWithHighestScores(word_score):
    results_list = sorted(word_score.items(), key=lambda x: (x[0]))
    results_list = sorted(results_list, key=lambda x: (x[1]), reverse=True)
    for i in range(min(10, len(results_list))):
        print(results_list[i][0], end='')
        if i + 1 < min(10, len(results_list)):
            print(", ", end='')
        else:
            print('')


# for each sentence calculates its value by counting top 10 words in it
def calculateSentenceScores(sentence_dict, word_score):
    for key in sentence_dict:
        total = []
        res = 0.0
        keys = word_tokenize(key)
        for word in keys:
            if word.isalnum():
                neo = stemming.stem(word)
                adding = word_score.get(neo, 0)
                total.append(adding)
        total.sort(reverse=True)
        numb = len(total)
        for i in range(min(10, numb)):
            res += total[i]
        sentence_dict[key] = res
    return sentence_dict


# print 5 (or less if sentence count is less than 5) sentences with highest scores - summary
def printSentencesWithHighestScores(sentence_dict):
    sentence_dict_list = list(sentence_dict.items())
    num = len(sentence_dict_list)
    selected_sentences = []
    for j in range(min(5, num)):
        maximum = 0
        index = -1
        for i in range(len(sentence_dict_list)):
            if sentence_dict_list[i][1] > maximum and i not in selected_sentences:
                maximum = sentence_dict_list[i][1]
                index = i
        selected_sentences.append(index)
    selected_sentences.sort()

    for i in range(min(5, len(sentence_dict_list))):
        print(sentence_dict_list[selected_sentences[i]][0], end='')
        if i < min(4, len(sentence_dict_list)):
            print(" ", end='')


def tf_idf_analysis(corpus, document_target):
    documents = getDocumentsInCorpus(corpus, False)
    if documents is None:
        print("Corpus not found")
        return

    word_occurrences_per_document, word_score, sentence_dict = initializeDictionaries(document_target, len(documents))
    if word_occurrences_per_document is None:
        print("Target document not found", end="\n\n")
        return

    word_occurrences_per_document = countWordOccurencesInCorpus(documents, word_occurrences_per_document)

    target_document_index = documents.index(document_target)
    word_score = calculateWordsScores(len(documents), target_document_index, word_occurrences_per_document, word_score)

    sys.stdout.reconfigure(encoding='utf-8')
    printWordsWithHighestScores(word_score)

    sentence_dict = calculateSentenceScores(sentence_dict, word_score)

    printSentencesWithHighestScores(sentence_dict)


if __name__ == "__main__":
    while True:
        corpus = input("\nCorpus path: ")
        document_target = input("Targeted document path: ")
        if corpus == "" or document_target == "":
            break
	print("\nResult:")
        tf_idf_analysis(corpus, document_target)
