from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings

# define columns
columns = {0: 'ID', 1: 'text', 2: 'empty_0', 3: 'pos', 4: 'empty_1',
           5: 'empty_2', 6: 'empty_3', 7: 'empty_4', 8: 'empty_5', 9: 'tox'}

# this is the folder in which train, test and dev files reside
data_folder = 'Data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='converted_data.conll',
                              test_file='converted_data.conll',
                              dev_file='converted_data.conll')

print(corpus.train[0].to_tagged_string('tox'))

# 2. what tag do we want to predict?
tag_type = 'tox'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

# load the model you trained
model = SequenceTagger.load('resources/taggers/example-pos/final-model.pt')

# create example sentence
sentence = Sentence('I hate all the fucking immigrants')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())