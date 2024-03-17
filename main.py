from tqdm import tqdm
from linguistics.corpus import Corpus

if __name__ == '__main__':
    # corpus = Corpus.from_files('D:/Corpora & Language Resources/modu-corenlp/layers-complete/*/*.json')
    corpus = Corpus.from_pickle('linguistics/corpus/corpus.pkl')

    for doc in tqdm(corpus.iter_documents()):
        for snt in doc.iter_sentences():
            for layer in corpus.layers:
                assert snt.get_annotation(layer)
