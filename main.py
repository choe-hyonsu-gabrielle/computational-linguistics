from transformers import AutoTokenizer
from linguistics.corpus import Corpus
from linguistics.corpus.process import POSProcess

if __name__ == '__main__':
    corpus = Corpus.from_files('D:/Corpora & Language Resources/modu-corenlp/layers-complete/*/*.json')
    corpus.to_pickle('linguistics/corpus/corpus.pkl')
    # corpus = Corpus.from_pickle('linguistics/corpus/corpus.pkl')

    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    pos_process = POSProcess(tokenizer=tokenizer)

    for snt in corpus.iter_sentences():
        alignment = pos_process(snt.pos)
        break