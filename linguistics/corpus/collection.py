import glob
import time
import random
from typing import Any, Union, Optional, Iterable
from collections import defaultdict
from tqdm import tqdm
from linguistics.corpus.layer import *
from linguistics.utils import load_json, save_pickle, load_pickle, timestamp


SENTENCE_LEVEL_LAYERS = {'pos': 'morpheme', 'wsd': 'WSD', 'ner': 'NE', 'el': 'NE', 'dep': 'DP', 'srl': 'SRL'}
DOCUMENT_LEVEL_LAYERS = {'za': 'ZA', 'cr': 'CR'}
DATATYPES_BY_LAYER = {
    'pos': POSLayer,
    'wsd': WSDLayer,
    'ner': NERLayer,
    'el': ELLayer,
    'dep': DEPLayer,
    'srl': SRLLayer,
    'za': ZALayer,
    'cr': CRLayer
}


class Annotations:
    def __init__(self, super_instance=None):
        self.super: Optional[Sentence] = super_instance
        self.pos: Optional[POSLayer] = None
        self.ner: Optional[NERLayer] = None
        self.el: Optional[ELLayer] = None
        self.wsd: Optional[WSDLayer] = None
        self.dep: Optional[DEPLayer] = None
        self.srl: Optional[SRLLayer] = None
        self.za: Optional[ZALayer] = None
        self.cr: Optional[CRLayer] = None

    def __repr__(self):
        return '\n\n'.join([f'`{_var}`: {_repr}' for _var, _repr in self.__dict__.items()])

    def get(self, layer: str, default: Any = None):
        assert layer != 'super'
        return getattr(self, layer, default)

    def add(self, layer: str, instance: Union[None, POSLayer, NERLayer, ELLayer, WSDLayer, DEPLayer, SRLLayer, ZALayer, CRLayer]):
        assert layer in self.__dict__ and layer != 'super'
        setattr(self, layer, instance)

    @property
    def ref_id(self):
        return self.super.ref_id

    @property
    def form(self):
        return self.super.canonical_form

    def word(self, word_id: int):
        return self.super.word_dict.get(word_id, None)


class Sentence:
    def __init__(self, snt_id: str, super_instance: Any = None):
        self.ref_id = snt_id
        self.forms = defaultdict(set)
        self.super: Optional[Document] = super_instance
        self.annotations = Annotations(self)
        self.index: dict[int, tuple[int, int]] = dict()  # mapping of {word_id: (begin, end)}
        self.word_dict = dict()   # mapping of {word_id: "word_form"}

    def __repr__(self):
        return f'<{self.__class__.__name__} → id: {self.ref_id}, form ({len(self.forms)}): "{self.canonical_form}">'

    def __len__(self):
        return len(self.canonical_form)

    @property
    def pos(self) -> POSLayer:
        return self.annotations.get('pos')

    @property
    def ner(self) -> NERLayer:
        return self.annotations.get('ner')

    @property
    def el(self) -> ELLayer:
        return self.annotations.get('el')

    @property
    def wsd(self) -> WSDLayer:
        return self.annotations.get('wsd')

    @property
    def dep(self) -> DEPLayer:
        return self.annotations.get('dep')

    @property
    def srl(self) -> SRLLayer:
        return self.annotations.get('srl')

    @property
    def za(self) -> ZALayer:
        """ only returns items relevant to the sentence
        :return: list of zero-anaphora items
        """
        if self.annotations.za is not None:
            return self.annotations.get('za')
        # filter by sentence_id in predicate to get rid of irrelevant items to current Sentence
        za_list = [za for za in self.super.doc_za if za.predicate.sentence_id == self.ref_id]
        za_dict = [dict(predicate=z.predicate.__dict__, antecedent=[a.__dict__ for a in z.antecedent]) for z in za_list]
        for za_item in za_dict:
            za_item['antecedent'] = [a for a in za_item['antecedent'] if a['sentence_id'] in ('-1', self.ref_id)]
        return ZALayer(layer='za', data=[za_item for za_item in za_dict if za_item['antecedent']], super_instance=self)

    @property
    def cr(self) -> CRLayer:
        """ only returns clusters of mentions relevant to the sentence
        :return: list of clustered mentions
        """
        if self.annotations.cr is not None:
            return self.annotations.get('cr')
        valid_clusters = []
        for cr_item in self.super.doc_cr:
            intra_sentence_coreference = [m.__dict__ for m in cr_item.mention if m.sentence_id == self.ref_id]
            if len(intra_sentence_coreference) > 1:
                valid_clusters.append(intra_sentence_coreference)
        return CRLayer(layer='cr', data=[dict(mention=cluster) for cluster in valid_clusters], super_instance=self)

    @property
    def words(self) -> list[str]:
        return list(self.word_dict.values())

    @property
    def canonical_form(self) -> str:
        if len(self.forms) == 1:
            return list(self.forms)[0]
        return sorted([(text, len(layers)) for text, layers in self.forms.items()], key=lambda x: x[1])[-1][0]

    def get_form(self, layer: str) -> Optional[str]:
        for form, layers in self.forms.items():
            if layer in layers:
                return form
        return None

    def add_form(self, form: str, layer: str):
        self.forms[form].add(layer)

    def add_word_index(self, words: dict):
        # it is supposed to be called once only if processing on `dep`
        assert not self.word_dict and not self.index
        self.word_dict = {w['id']: w['form'] for w in words}
        self.index = {w['id']: (w['begin'], w['end']) for w in words}

    def word_id_to_span_ids(self, word_id: int) -> Optional[int]:
        return self.index.get(word_id, None)

    def span_ids_to_word_id(self, begin: int, end: int) -> tuple[int, int]:
        assert begin <= end
        word_begin, word_end = None, None
        for word_id, (span_begin, span_end) in self.index.items():
            if span_begin <= begin <= end <= span_end:
                return word_id, word_id
            if span_begin <= begin:
                word_begin = word_id
                continue
            if end <= span_end:
                word_end = word_id
                break
        return word_begin, word_end

    def doc_to_snt_annotation(self):
        if 'za' in self.super.annotations:
            self.annotations.add('za', self.za)
        if 'cr' in self.super.annotations:
            self.annotations.add('cr', self.cr)

    def get_annotation(self, layer: str) -> Union[POSLayer, NERLayer, ELLayer, WSDLayer, DEPLayer, SRLLayer, CRLayer, ZALayer]:
        return self.annotations.get(layer, None)

    def add_annotation(self, layer: str, data: Any):
        assert layer in SENTENCE_LEVEL_LAYERS
        self.annotations.add(layer, DATATYPES_BY_LAYER[layer](layer=layer, data=data, super_instance=self))


class Document:
    def __init__(self, doc_id: str, super_instance=None):
        self.ref_id = doc_id
        self.sentences = defaultdict(Sentence)
        self.super: Optional[Corpus] = super_instance
        self.annotations = defaultdict(Layer)

    def __len__(self):
        return len(self.sentences)

    def get_sentence(self, snt_id: str) -> Sentence:
        return self.sentences.get(snt_id, None)

    def add_sentence(self, snt_id: str, instance: Sentence):
        self.sentences[snt_id] = instance

    def get_annotation(self, layer: str) -> Layer:
        assert layer in DOCUMENT_LEVEL_LAYERS
        return self.annotations.get(layer, None)

    def add_annotation(self, layer: str, data: Any):
        assert layer in DOCUMENT_LEVEL_LAYERS
        self.annotations[layer] = DATATYPES_BY_LAYER[layer](layer=layer, data=data, super_instance=self)

    def iter_sentences(self) -> Iterable[Sentence]:
        return (self.get_sentence(snt_id) for snt_id in self.sentences.keys())

    @property
    def doc_cr(self) -> list[CRItem]:
        return self.get_annotation('cr').data

    @property
    def doc_za(self) -> list[ZAItem]:
        return self.get_annotation('za').data


class Corpus:
    def __init__(self):
        self.dirs = dict()
        self.layers = set()
        self.documents = defaultdict(Document)
        self.index = defaultdict(str)  # mapping of snt_id to doc_id
        self.update = None

    def __len__(self):
        return len(self.index)  # length of all sentences

    def __repr__(self):
        return f'<{self.__class__.__name__} → items: {len(self)}, updated: {self.update}, layers: {tuple(self.layers)}>'

    def get_document(self, doc_id: str) -> Document:
        return self.documents.get(doc_id, None)

    def add_document(self, doc_id: str, instance: Document):
        self.documents[doc_id] = instance

    def get_sentence(self, snt_id: str) -> Sentence:
        doc_id = self.index.get(snt_id, None)
        if doc_id:
            document: Document = self.documents[doc_id]
            return document.get_sentence(snt_id)
        return None

    def add_sentence(self, snt_id: str, instance: Sentence, doc_id: str):
        self.documents[doc_id].add_sentence(snt_id, instance)

    def iter_documents(self) -> Iterable[Document]:
        return (self.get_document(doc_id) for doc_id in self.documents.keys())

    def iter_sentences(self) -> Iterable[Sentence]:
        return (self.get_sentence(snt_id) for snt_id in self.index)

    def filter_by(
            self,
            len_range: Optional[tuple[int, int]] = None,
            include: Optional[Union[str, list]] = None,
            exclude: Optional[Union[str, list]] = None,
            startswith: Optional[Union[str, list]] = None,
            endswith:  Optional[Union[str, list]] = None,
            random_state: int = None
    ):
        pool = list(self.index)
        if random_state:
            random.seed(random_state)
            random.shuffle(pool)
        for snt_id in pool:
            sentence = self.get_sentence(snt_id).canonical_form
            valid = True
            valid = len_range[0] <= len(sentence) <= len_range[-1] and valid if len_range else valid
            valid = any([exp in sentence for exp in list(include)]) and valid if include else valid
            valid = not any([exp in sentence for exp in list(exclude)]) and valid if exclude else valid
            valid = any([sentence.startswith(prefix) for prefix in list(startswith)]) and valid if startswith else valid
            valid = any([sentence.endswith(suffix) for suffix in list(endswith)]) and valid if endswith else valid
            if valid:
                yield self.get_sentence(snt_id)

    def sample_documents(self, k: int, random_state: int = None):
        assert k <= len(self.documents)
        random.seed(random_state)
        return [document for document in random.sample(population=list(self.documents), k=k)]

    def sample_sentences(self, k: int, random_state: int = None):
        assert k <= len(self.index)
        random.seed(random_state)
        sample_ids = random.sample(population=list(self.index), k=k)
        return [self.get_sentence(snt_id) for snt_id in sample_ids]

    @staticmethod
    def from_files(filepath: str):
        target_files = {layer.split('\\')[-2]: layer for layer in glob.glob(filepath)}
        corpus = Corpus()
        corpus.dirs.update(target_files)
        corpus.layers.update(target_files)
        for layer, filename in tqdm(corpus.dirs.items(), desc=f'- loading {len(corpus.dirs)} corpora files from scratch'):
            for _doc in load_json(filename)['document']:
                doc_id = _doc['id']
                if doc_id not in corpus.documents:
                    corpus.documents[doc_id] = Document(doc_id=doc_id, super_instance=corpus)
                document = corpus.get_document(doc_id=doc_id)
                if layer in SENTENCE_LEVEL_LAYERS:
                    for _snt in _doc['sentence']:
                        snt_id = _snt['id']
                        form = _snt['form']
                        if snt_id not in document.sentences:
                            document.sentences[snt_id] = Sentence(snt_id=snt_id, super_instance=document)
                        sentence = document.get_sentence(snt_id=snt_id)
                        sentence.add_form(form=form, layer=layer)
                        data = _snt[SENTENCE_LEVEL_LAYERS[layer]]
                        sentence.add_annotation(layer=layer, data=data)
                        if layer == 'dep' and 'word' in _snt:
                            sentence.add_word_index(_snt['word'])
                        corpus.index[snt_id] = doc_id
                elif layer in DOCUMENT_LEVEL_LAYERS:
                    data = _doc[DOCUMENT_LEVEL_LAYERS[layer]]
                    document.add_annotation(layer=layer, data=data)
                    for _snt in _doc['sentence']:
                        snt_id = _snt['id']
                        form = _snt['form']
                        if snt_id not in document.sentences:
                            document.sentences[snt_id] = Sentence(snt_id=snt_id, super_instance=document)
                        sentence = document.get_sentence(snt_id=snt_id)
                        sentence.add_form(form=form, layer=layer)
                        corpus.index[snt_id] = doc_id
                        sentence.doc_to_snt_annotation()
                else:
                    raise ValueError(f'`{layer}` is unsupported annotation type: {list(DATATYPES_BY_LAYER)}')
        # timestamp
        corpus.update = timestamp()
        return corpus

    @staticmethod
    def from_pickle(filename):
        print(f'- loading corpus from `{filename}` file. it takes more or less than 1 minute average.')
        start = time.time()
        corpus: Corpus = load_pickle(filename)
        lapse = time.time() - start
        print(f'- unpickling `{filename}`, {lapse:.2f} sec lapsed: {corpus}')
        return corpus

    def to_pickle(self, filename):
        # timestamp
        self.update = timestamp()
        save_pickle(filename=filename, instance=self)
