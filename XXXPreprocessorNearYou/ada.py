with open('/usr/local/src/kms.py', 'rb') as source_file:
    code = compile(source_file.read(), '/usr/local/src/kms.py', 'exec')
    exec(code)

# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

import keras_nlp
import keras

keras.mixed_precision.set_global_policy("mixed_float16")

bsize = 16
imdb_train = keras.utils.text_dataset_from_directory(dpath + "aclImdb/train", batch_size=bsize)
imdb_test = keras.utils.text_dataset_from_directory(dpath + "aclImdb/test", batch_size=bsize)

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased", num_classes=2)
classifier.fit(imdb_train, validation_data=imdb_test, epochs=1)
classifier.save('a.keras')
