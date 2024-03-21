from transformers import AutoTokenizer
from linguistics.corpus import Corpus
from linguistics.corpus.process.pos import POSProcess

if __name__ == '__main__':
    # corpus = Corpus.from_files('D:/Corpora & Language Resources/modu-corenlp/layers-complete/*/*.json')
    # corpus.to_pickle('linguistics/corpus/corpus.pkl')
    corpus = Corpus.from_pickle('linguistics/corpus/corpus.pkl')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    pos_process = POSProcess(tokenizer=tokenizer)

    while not input('press enter to continue:'):
        print()
        for snt in corpus.sample_sentences(k=500):
            if snt.pos:
                alignment = pos_process(snt.pos)
                alignment.align()
        print()
