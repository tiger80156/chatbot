# import numpy as np
# import tensorflow as tf
import re
import time


def re_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[#$\/*()@^+={}~,;.?|<>-]", "", text)
    return text


def dataPreprocessing():
    # Read the corups file
    lines = open("./movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
    conversations = open("./movie_conversations.txt", encoding='utf-8', errors='ignore').read().split('\n')

    # Extract the converastion
    id2Line = {}
    for line in lines:
        _line = line.split(" +++$+++ ")

        if len(_line) == 5:
            id2Line[_line[0]] = _line[4]

    # Get the conversation id
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))

    questions = []
    answers = []

    # Get the answers and questions list from dialogue
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2Line[conversation[i]])
            answers.append(id2Line[conversation[i + 1]])

    # re_text the element the the questions list and answers list
    questions_clean = []
    answers_clean = []

    for question in questions:
        questions_clean.append(re_text(question))
    for answer in answers:
        answers_clean.append(re_text(answer))

    return questions_clean, answers_clean

# filter out the word frequent < 15


def filterTheNonFrequentWord(questions, answers, threshold=15):

    wordCount = {}

    for question in questions:
        for word in question.split():
            if word not in wordCount:
                wordCount[word] = 1
            else:
                wordCount[word] += 1

    for answer in answers:
        for word in answer.split():
            if word not in wordCount:
                wordCount[word] = 1
            else:
                wordCount[word] += 1

    # Create the quesiotn word ID
    questionWord2Count = {}
    wordID = 0

    for word, count in wordCount.items():

        if count >= threshold:
            questionWord2Count[word] = wordID
            wordID += 1

    # Create the  answer word ID
    answerWord2Count = {}
    wordID = 0

    for word, count in wordCount.items():

        if count >= threshold:
            answerWord2Count[word] = wordID
            wordID += 1

    tokens = ['<EOS>', '<SOS>', '<PAD>', '<OUT>']

    for token in tokens:
        answerWord2Count[token] = len(answerWord2Count)
        questionWord2Count[token] = len(questionWord2Count)

    questionCount2word = {count: word for word, count in questionWord2Count.items()}

    return questionWord2Count, answerWord2Count, questionCount2word


def addToken(answers):

    for i in range(len(answers)):
        answers[i] += " <EOS>"

    return answers


# Sentence Embedding the word in the question and answer sentence to word ID.
def sentenceEncoding(questions, answers, questionsWord2Id, answerWord2Id):
    questionsEncoding = []
    answersEncoding = []

    for question in questions:

        encoding = []
        for word in question.split():

            if word not in questionsWord2Id:
                encoding.append(questionsWord2Id["<OUT>"])

            else:
                encoding.append(questionsWord2Id[word])

        questionsEncoding.append(encoding)

    for answer in answers:
        encoding = []

        for word in answer.split():
            if word not in answerWord2Id:
                encoding.append(answerWord2Id["<OUT>"])

            else:
                encoding.append(answerWord2Id[word])

        answersEncoding.append(encoding)

    return questionsEncoding, answersEncoding

# Sort the training data by length and filter out the sectence size bigger than 25


def sortByLen(questions, answers):
    # print(questions[:5])
    sortedQuestions = []
    sortedAnswers = []

    for length in range(1, 25 + 1):
        for i in enumerate(questions):
            if len(i[1]) == length:
                sortedQuestions.append(questions[i[0]])
                sortedAnswers.append(answers[i[0]])

    return sortedQuestions, sortedAnswers
