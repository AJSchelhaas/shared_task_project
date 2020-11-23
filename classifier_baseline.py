# Basic baseline classifier for toxic spans detection
# File name: classifier_baseline.py
# Date: 08-10-2020

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import sys


def main(train=False, predict_file=None):
    # Train if required
    if train:
        train_model('Data')

    # load the model you trained
    model = SequenceTagger.load('resources/taggers/toxic_classifier/final-model.pt')

    # Predict
    results = []
    if predict_file is None:
        result = predict(model, "I really think you are a dumb motherfucking idiot!")
        results.append(result)
    else:
        with open(predict_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result = predict(model, line)
                results.append(result)

    #show results
    for result in results:
        print(result)

    write_results(results, "results_baseline.csv")


def train_model(directory='Data'):
    # define columns
    columns = {0: 'ID', 1: 'text', 2: 'empty_0', 3: 'pos', 4: 'empty_1',
               5: 'empty_2', 6: 'empty_3', 7: 'empty_4', 8: 'empty_5', 9: 'tox'}

    # this is the folder in which train, test and dev files reside
    data_folder = directory

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                    train_file='converted_data_train.conll',
                                    test_file='converted_data_test.conll',
                                    dev_file='converted_data_dev.conll')

    # tag to predict
    tag_type = 'tox'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # embeddings
    embedding_types = [WordEmbeddings('glove')]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # start training
    trainer.train('resources/taggers/toxic_classifier',
                    learning_rate=0.1,
                    mini_batch_size=32,
                    max_epochs=5)


def predict(model, predict_sentence):
    sentence = Sentence(predict_sentence)
    model.predict(sentence)
    print(predict_sentence)

    dic = sentence.to_dict(tag_type='tox')
    toxic_spans = []
    for token in dic['entities']:
        label = int(token['labels'][0].value)
        if label == 1:
            start_pos = token['start_pos']
            end_pos = token['end_pos']
            for i in range(start_pos, end_pos):
                toxic_spans.append(i)

    return [toxic_spans, predict_sentence]


def write_results(results, filename):
    with open(filename, "w") as f:
        for result in results:
            result_string = '"' + str(result[0]) + '","' + str(result[1]) + '"'
            f.write(result_string)


if __name__ == "__main__":
    train_mode = False
    if len(sys.argv) > 1:
        train_mode = sys.argv[1] == "1"

    predict_file = None
    if len(sys.argv) > 2:
        predict_file = sys.argv[2]

    main(train_mode, predict_file)