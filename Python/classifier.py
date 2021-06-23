import glob
import string
import math
import pandas as pd
import scipy
import numpy as np
import nltk
import nltk.sentiment
import nltk.corpus
import sklearn
import sklearn.tree
import timeit

# Train Bag-of-Words Model
def train_BOW(file_name, source_data):
    bag_of_words = {}
    for review in source_data:
        with open(review, 'r') as f:
            line = f.read()
            line = clean_list(line)
            for word in line:
                bag_of_words[word] = bag_of_words.get(word, 0) + 1
    with open(file_name, 'w') as f:
        for _ in sorted(bag_of_words, key=bag_of_words.get, reverse=True):
            f.write("%s %d\n" % (_, bag_of_words[_]))
    return bag_of_words

# Load file and process Bag-of-Words dictionary.
def load_BOW(file_name):
    bag_of_words = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, count = line.split(" ")
            bag_of_words[key] = int(count)

    return bag_of_words

# Parsing lists
def list_parse(pos_list, neg_list):
    p_word_list = {}
    n_word_list = {}
    for key in neg_list.keys():
        if key in pos_list.keys():
            if int(pos_list[key]) >= int(neg_list[key]):
                p_word_list[key] = pos_list[key]
            elif int(neg_list[key]) > int(pos_list[key]):
                n_word_list[key] = neg_list[key]
        else:
            n_word_list[key] = neg_list[key]
    for key in pos_list.keys():
        if key not in neg_list.keys():
            p_word_list[key] = pos_list[key]
    smaller_list_len=min(len(n_word_list),len(p_word_list))
    sorted_neg_list = sorted(n_word_list, key=n_word_list.get, reverse=True)[:smaller_list_len] #Sorts 
    sorted_pos_list = sorted(p_word_list, key=p_word_list.get, reverse=True)[:smaller_list_len]
    clean_neg={k:n_word_list[k] for k in sorted_neg_list}
    clean_pos = {k: p_word_list[k] for k in sorted_pos_list}
    return clean_pos, clean_neg

# Cleaning up lines
def clean_list(line):
    line = line.split(" ")
    if ft1:
        line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
    if ft2:
        line = [word for word in line if word.isalpha()]
    if ft3:
        line = [word for word in line if len(word) > 1]
    if ft4:
        line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
    return line


def tokenizer(line):
    line = clean_list(line)
    return(line)

# Helper function for processing
def bow_helper(pos, neg, _file_name):
    score = 0
    with open(_file_name, 'r') as f:
        line = f.read()
        line = clean_list(line)
        for word in line:
            increment = 1
            if word in neg:
                score -= increment
            if word in pos:
                score += increment
    if score >= 0:
        return 1
    else:
        return 0


# Helper function for Naive Bayes
def nb_Helper(pos_bow, neg_bow,neg_size,pos_size, file_name):
    neg_statistic = 0
    pos_statistic = 0
    with open(file_name, 'r') as f:
        line = f.read()
        line = clean_list(line)
        for word in line:
            if word in neg_bow.keys():
                neg_statistic += math.log(int(neg_bow[word])/neg_size)
            else:
                neg_statistic += math.log(1/neg_size)
            if word in pos_bow.keys():
                pos_statistic += math.log(int(pos_bow[word])/pos_size)
            else:
                pos_statistic += math.log(1/pos_size)

    if neg_statistic > pos_statistic:
        return 0
    else:
        return 1

# Naive Bayes
def nb(pos_keys, neg_keys, pos_test_data, neg_test_data):
    print('Running the Naive Bayes Classifier')
    neg_l = len(neg_BOW)
    pos_l = len(pos_BOW)
    neg_score = 0
    pos_score = 0
    neg_size = sum(neg_keys.values()) + len(neg_keys.keys())
    pos_size = sum(pos_keys.values()) + len(pos_keys.keys())
    for file in neg_test_data[500:]:
        neg_score += nb_Helper(pos_keys, neg_keys, neg_size, pos_size, file)
    print('Negative Score: '+ str((500-neg_score)/500))
    for file in pos_test_reviews[500:]:
        pos_score += nb_Helper(pos_keys, neg_keys, neg_size, pos_size, file)
    print('Positive Score: ' + str(pos_score/500))

# Decision Tree
def dt(pos_keys, neg_keys, basis, train_stat, test_stat):
    neg_l = len(neg_BOW)
    pos_l = len(pos_BOW)
    print('Decision Tree Classifier')
    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(train_stat,basis)
    scores = clf.predict(test_stat)
    neg_score = 0
    pos_score = 0
    negative_score = 0
    positive_score = 0
    for i in range(0, 250):
        negative_score += scores[i]
    for i in range(250, 500):
        positive_score += scores[i]


    print('Negative Score: ' + str((500 - negative_score) / 500))
    print('Positive Score: ' + str(positive_score/500))

# Logistic Regression
def lr(pos_keys, neg_keys, basis, train_stat, test_stat):
    neg_l = len(neg_BOW)
    pos_l = len(pos_BOW)
    print('Logistic Regression Classifier')
    clf = sklearn.linear_model.SGDClassifier()
    clf.fit(train_stat,basis)
    scores = clf.predict(test_stat)
    neg_score = 0
    pos_score = 0
    negative_score = 0
    positive_score = 0
    for i in range(0, 500):
        negative_score += scores[i]
    for i in range(500, 1000):
        positive_score += scores[i]


    print('Negative Score: ' + str((500 - negative_score) / 500))
    print('Positive Score: ' + str(positive_score/500))


# Main function, long because of the way "tests" are setup
if __name__ == "__main__":
    
    # Sort all files in pos and neg directories
    pos_test_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/pos/*'))
    neg_test_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/neg/*'))
    
    # print(pos_test_reviews)
    # print(neg_test_reviews)
    positive_file_name = "positive_BOW.txt"
    negative_file_name = "negative_BOW.txt"
    
    ft1 = False
    ft2 = False
    ft3 = False
    ft4 = False

    # Prompt users on feature selection
    prompt = input("Feature 1: Remove Punctuation? Y/N")
    if prompt == "Y" or prompt == "y":
    	ft1 = True
    prompt = input("Feature 2: Remove non-Alphanumeric Characters? Y/N")
    if prompt == "Y" or prompt == "y":
    	ft2 = True

    prompt = input("Feature 3: Filter out short tokens? Y/N")
    if prompt == "Y" or prompt == "y":
    	ft3 = True 

    prompt = input("Feature 4: Remove Stopwords? Y/N")
    if prompt == "Y" or prompt == "y":
    	ft4 = True 
    

    # Uncomment this section to generate new data
    # neg_BOW = train_BOW(negative_file_name, neg_test_reviews, ft1, ft2, ft3, ft4) #Uncomment this part to generate new data
    # pos_BOW = train_BOW(positive_file_name, pos_test_reviews, ft1, ft2, ft3, ft4) #Uncomment this part ot generate new data

    
    # Uncomment this section to load BOW model
    neg_BOW = load_BOW(negative_file_name)
    pos_BOW = load_BOW(positive_file_name)

    pos_keys, neg_keys = list_parse(pos_BOW, neg_BOW)

    # Begin BOW
    print('Bag of Words')
    start = timeit.default_timer()
    max_neg = 0
    max_pos = 0
    temp = 0

    for i in neg_keys:
        test=int(neg_keys[i])
        if test>temp:
            temp=test
        max_neg += test
    max_neg /= temp
    temp = 0
    for i in pos_keys:
        test=int(pos_keys[i])
        if test>temp:
            temp=test
        max_pos += test
    max_pos /= temp

    negative_score = 0
    positive_score = 0
    for file in neg_test_reviews[500:]:
        negative_score += bow_helper(pos_keys, neg_keys,  file)
    print('Negative Score: '+ str((500-negative_score)/500))
    for file in pos_test_reviews[500:]:
        positive_score += bow_helper(pos_keys, neg_keys, file)
    print('Positive Score: ' + str(positive_score/500))
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    # Naive Bayes
    start = timeit.default_timer()
    nb(pos_keys, neg_keys, pos_test_reviews, neg_test_reviews)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    # Pre-processing for SKLearn for Logistic Regression and Decision Tree classifiers.
    start = timeit.default_timer()
    combined_keys = []
    train = neg_test_reviews[:500]+pos_test_reviews[:500]
    test = neg_test_reviews[500:]+pos_test_reviews[500:]
    # print(test)
    for key in pos_keys:
    	combined_keys.append(key)
    for key in neg_keys:
    	combined_keys.append(key)
    combined_keys = sorted(list(set(combined_keys)))
    basis = [0] * 500 + [1] * 500
    vector = sklearn.feature_extraction.text.CountVectorizer(input='filename', tokenizer=tokenizer,
                                                                 vocabulary=combined_keys)
    train_s = vector.fit_transform(train)
    test_s = vector.fit_transform(test)

    # Decision Tree
    dt(pos_keys, neg_keys, basis, train_s, test_s)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    start = timeit.default_timer()
    # Logistic Regression 
    lr(pos_keys, neg_keys, basis, train_s, test_s)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  # understandably quicker since I'm pre-processing for both

