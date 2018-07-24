def get_conversations_structure():
    with open("../Downloads/cornell_movie_dialogue_corpus/movie_conversations.txt") as file:
        conv_line = file.read().split("\n")
        conv = []
        for line in conv_line[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            conv.append(_line.split(','))

        return conv

def get_sentences():
    with open("../Downloads/cornell_movie_dialogue_corpus/movie_lines.txt", encoding = 'iso8859_15') as file:
        conv_line = file.read().split("\n")
        conv_sentence = {}

        for line in conv_line:
            _line = line.split(" +++$+++ ")
            conv_sentence[_line[0]] = _line[-1]

        return conv_sentence


def get_conversations():
    dialogs = get_conversations_structure()
    con_sentence = get_sentences()

    movie_dialogs = []

    for dialog in dialogs:
        sentence = []

        for sentenceId in dialog:
            sentence.append(con_sentence[sentenceId])

        movie_dialogs.append(sentence)

    return movie_dialogs


