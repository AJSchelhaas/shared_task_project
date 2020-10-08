from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings

# define columns
columns = {0: 'text', 1: 'tox'}

# this is the folder in which train, test and dev files reside
data_folder = ''

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train',
                              test_file='test',
                              dev_file='dev')

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