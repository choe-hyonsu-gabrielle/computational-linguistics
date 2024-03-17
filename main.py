from tqdm import tqdm
from linguistics.corpus import Corpus
from collections import defaultdict

if __name__ == '__main__':
    # corpus = Corpus.from_files('D:/Corpora & Language Resources/modu-corenlp/layers-complete/*/*.json')
    corpus = Corpus.from_pickle('linguistics/corpus/corpus.pkl')
    ordered_layers = ['pos', 'dep', 'wsd', 'srl', 'ner', 'el', 'za', 'cr']

    counts = defaultdict(int)
    available = defaultdict(int)

    for s in tqdm(corpus.iter_sentences()):
        activated = {layer: bool(len(s.get_annotation(layer))) for layer in ordered_layers}
        for k, v in activated.items():
            if v:
                available[k] += 1
        key = '-'.join([k for k, v in activated.items() if v])
        counts[key] += 1

        # if sum(activated.values()) == 8:
        #     for lay in ordered_layers:
        #         print(s.get_annotation(lay))
        #         print()
        #     break

    for k, v in sorted(counts.items(), key=lambda x: len(x[0]), reverse=True):
        print(f'- `{k}`: {v}')

    for k in ordered_layers:
        print(f'- `{k}`: {available[k]}')
