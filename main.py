
import sys
sys.path.append("..")
from common.download_utils import download_week3_resources
download_week3_resources()
import gensim
from grader import Grader
grader = Grader()
from gensim.models.keyedvectors import KeyedVectors

wv_embeddings = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)


def check_embeddings(embeddings):
    error_text = "Something wrong with your embeddings ('%s test isn't correct)."
    most_similar = embeddings.most_similar(positive=['woman', 'king'], negative=['man'])
    if len(most_similar) < 1 or most_similar[0][0] != 'queen':
        return error_text % "Most similar"

    doesnt_match = embeddings.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch'])
    if doesnt_match != 'cereal':
        return error_text % "Doesn't match"

    most_similar_to_given = embeddings.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse'])
    if most_similar_to_given != 'sound':
        return error_text % "Most similar to given"

    return "These embeddings look good."


print(check_embeddings(wv_embeddings))

import numpy as np

def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    words_embedding = [embeddings[word] for word in question.split() if word in embeddings]
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding)
    return words_embedding.mean(axis=0)

def question_to_vec_tests():
    if (np.zeros(300) != question_to_vec('', wv_embeddings)).any():
        return "You need to return zero vector for empty question."
    if (np.zeros(300) != question_to_vec('thereisnosuchword', wv_embeddings)).any():
        return "You need to return zero vector for the question, which consists only unknown words."
    if (wv_embeddings['word'] != question_to_vec('word', wv_embeddings)).any():
        return "You need to check the corectness of your function."
    if ((wv_embeddings['I'] + wv_embeddings['am']) / 2 != question_to_vec('I am', wv_embeddings)).any():
        return "Your function should calculate a mean of word vectors."
    if (wv_embeddings['word'] != question_to_vec('thereisnosuchword word', wv_embeddings)).any():
        return "You should not consider words which embeddings are unknown."
    return "Basic tests are passed."

print(question_to_vec_tests())

import nltk
nltk.download('stopwords')
from util import array_to_string

question2vec_result = []
for question in open('data/test_embeddings.tsv'):
    question = question.strip()
    answer = question_to_vec(question, wv_embeddings)
    question2vec_result = np.append(question2vec_result, answer)

grader.submit_tag('Question2Vec', array_to_string(question2vec_result))

def hits_count(dup_ranks, k):
    """
        dup_ranks: list of duplicates' ranks; one rank per question;
                   length is a number of questions which we are looking for duplicates;
                   rank is a number from 1 to len(candidates of the question);
                   e.g. [2, 3] means that the first duplicate has the rank 2, the second one — 3.
        k: number of top-ranked elements (k in Hits@k metric)

        result: return Hits@k value for current ranking
    """
    count = 0
    for rank in dup_ranks:
        if rank <= k:
            count += 1
    return count / (len(dup_ranks) + 1e-8)


def test_hits():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1, 1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function."

    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown",
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."

    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."

print(test_hits())

def dcg_score(dup_ranks, k):
    """
        dup_ranks: list with ranks for each duplicate (the best rank is 1, the worst is len(dup_ranks))
        k: number of top-ranked elements

        result: float number
    """
    score = 0
    for rank in dup_ranks:
        if rank <= k:
            score += 1/np.log2(1+rank)
    return score/(len(dup_ranks)+1e-8)



def test_dcg():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1 / (np.log2(3)), 1 / (np.log2(3)), 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function."

    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown",
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."

    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, (1 + (1 / (np.log2(3)))) / 2]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."

print(test_dcg())

test_examples = [
    [1],
    [1, 2],
    [2, 1],
    [1, 2, 3],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [9, 5, 4, 2, 8, 10, 7, 6, 1, 3],
    [4, 3, 5, 1, 9, 10, 7, 8, 2, 6],
    [5, 1, 7, 6, 2, 3, 8, 9, 10, 4],
    [6, 3, 1, 4, 7, 2, 9, 8, 10, 5],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
]

hits_results = []
for example in test_examples:
    for k in range(len(example)):
        hits_results.append(hits_count(example, k + 1))
grader.submit_tag('HitsCount', array_to_string(hits_results))


dcg_results = []
for example in test_examples:
    for k in range(len(example)):
        dcg_results.append(dcg_score(example, k + 1))
grader.submit_tag('DCGScore', array_to_string(dcg_results))

def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data

validation = read_corpus('data/validation.tsv')

from sklearn.metrics.pairwise import cosine_similarity


def rank_candidates(question, candidates, embeddings, dim=300):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings

        result: a list of pairs (initial position in the list, question)
    """

    vecq = np.array([np.array(question_to_vec(question, embeddings, dim))])
    vecc = np.array([np.array(question_to_vec(can, embeddings, dim)) for can in candidates])
    scores = list(cosine_similarity(vecq, vecc)[0])
    tl = [(i, candidates[i], scores[i]) for i in range(len(candidates))]
    stl = sorted(tl, key=lambda x: x[2], reverse=True)
    return [(t[0], t[1]) for t in stl]


def test_rank_candidates():
    questions = ['converting string to list', 'Sending array via Ajax fails']
    candidates = [['Convert Google results object (pure js) to Python object',
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'],
                  ['Getting all list items of an unordered list in PHP',
                   'WPF- How to update the changes in list item of a list',
                   'select2 not displaying search results']]
    results = [[(1, 'C# create cookie from string and send it'),
                (0, 'Convert Google results object (pure js) to Python object'),
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'),
                (2, 'select2 not displaying search results'),
                (1, 'WPF- How to update the changes in list item of a list')]]
    for question, q_candidates, result in zip(questions, candidates, results):
        ranks = rank_candidates(question, q_candidates, wv_embeddings, 300)
        if not np.all(ranks == result):
            return "Check the function."
    return "Basic tests are passed."

print(test_rank_candidates())

wv_ranking = []
for line in validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_ranking.append([r[0] for r in ranks].index(0) + 1)

for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_ranking, k), k, hits_count(wv_ranking, k)))

for line in validation[:3]:
    q, *examples = line
    print(q, *examples[:3])

from util import text_prepare

prepared_validation = []
for line in validation:
    q, *ex = line
    q = text_prepare(q)
    for i,e in enumerate(ex):
        ex[i] = text_prepare(e)
    prepared_validation.append([q,*ex])

wv_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)

for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_prepared_ranking, k),
                                              k, hits_count(wv_prepared_ranking, k)))

def prepare_file(in_, out_):
    out = open(out_, 'w')
    for line in open(in_, encoding='utf8'):
        line = line.strip().split('\t')
        new_line = [text_prepare(q) for q in line]
        print(*new_line, sep='\t', file=out)
    out.close()

prepare_file('./data/validation.tsv', './data/tp_v.tsv')
prepare_file('./data/test.tsv', './data/tp_t.tsv')
prepare_file('./data/train.tsv', './data/tp_train.tsv')

from util import matrix_to_string

w2v_ranks_results = []
prepared_test_data = './data/tp_t.tsv'
for line in open(prepared_test_data):
    q, *ex = line.strip().split('\t')
    ranks = rank_candidates(q, ex, wv_embeddings, 300)
    ranked_candidates = [r[0] for r in ranks]
    w2v_ranks_results.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])

grader.submit_tag('W2VTokenizedRanks', matrix_to_string(w2v_ranks_results))

starspace_embeddings = {}
for line in open('./data/sodd.tsv'):
    word,*vec = line.strip().split()
    vf = []
    for v in vec:
        # print(v)
        vf.append(float(v))
    starspace_embeddings[word] = np.array(vf)

ss_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, starspace_embeddings, 100)
    ss_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)

for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(ss_prepared_ranking, k),
                                               k, hits_count(ss_prepared_ranking, k)))

starspace_ranks_results = []
prepared_test_data = './data/tp_t.tsv'
for line in open(prepared_test_data):
    q, *ex = line.strip().split('\t')
    ranks = rank_candidates(q, ex, starspace_embeddings, 100)
    ranked_candidates = [r[0] for r in ranks]
    starspace_ranks_results.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])

grader.submit_tag('StarSpaceRanks', matrix_to_string(starspace_ranks_results))

