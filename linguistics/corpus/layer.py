import pprint
from linguistics.utils import subgroups


class Item:
    """ super class for individual label or sub-structural items in annotation layer """
    def __repr__(self):
        return f'<{self.__class__.__name__}→{self.__dict__}>'


class Layer:
    def __init__(self, layer: str, data: list, super_instance=None):
        """ super class for annotation layer
        :param layer: an abbreviation of layer (ex. "pos", "dep", "srl", ...)
        :param data: list object of actual data
        :param super_instance: instance of Sentence or Document
        """
        self.super = super_instance
        self.layer = layer
        self.data = data

    def __repr__(self):
        name_of_class = self.__class__.__name__
        name_of_super = self.super.__class__.__name__
        header = f'<{name_of_class} → id: "{self.super.ref_id}" ({name_of_super}), items: {len(self.data)}>'
        pformat = pprint.pformat(self.tolist(), indent=2)
        return '\n'.join([header, pformat])

    def tolist(self):
        return self.data


class POSItem(Item):
    def __init__(self, **kwargs):
        self.id: int = kwargs['id']
        self.form: str = kwargs['form']
        self.label: str = kwargs['label']
        self.word_id: int = kwargs['word_id']
        self.position: int = kwargs['position']


class POSLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[POSItem(**d) for d in data], super_instance=super_instance)

    def tostring(self):
        """
        returns serialized POS tagged sequence
        ex. "가장/MAG 크/VA+ㄴ/ETM 충격/NNG+을/JKO 받/VV+는/ETM 곳/NNG+은/JX 관광업/NNG+계/XSN+이/VCP+다/EF+./SF"
        :return: str
        """
        words = []
        for word in subgroups(items=self.data, by='word_id', starts_from=1):
            words.append('+'.join(['/'.join([token.form, token.label]) for token in word]))
        return ' '.join(words).strip()

    def tolist(self) -> list[POSItem]:
        """
        :return: list of POSItem
        """
        return self.data

    def numbered_items(self) -> list[tuple[int, str]]:
        """
        returns numbered tuples starts with 1
        ex. [(1, '가장/MAG'), (2, '크/VA+ㄴ/ETM'), (3, '충격/NNG+을/JKO'), (4, '받/VV+는/ETM'), ...]
        :return: list of tuples (int, str) or individual item of (int, str)
        """
        return [(n+1, w) for n, w in enumerate(self.tostring().split())]

    def make_lemma_form(self, index: int) -> str:
        word_pos: list[POSItem] = subgroups(items=self.data, by='word_id', starts_from=1)[index-1]
        word_pos: list[tuple[str, str]] = [(i.form, i.label) for i in word_pos]
        contents = []
        for form, label in word_pos:
            if label in ('NNG', 'NA', 'NF', 'NP', 'SL', 'VV', 'VA', 'VCN', 'MAG', 'MMA', 'XSV', 'XSA', 'XSN', 'XR'):
                contents.append(form)
        return ''.join(contents)


class NERItem(Item):
    def __init__(self, **kwargs):
        self.id: int = kwargs['id']
        self.form: str = kwargs['form']
        self.label: str = kwargs['label']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']


class NERLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[NERItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[NERItem]:
        return self.data


class ELItem(Item):
    def __init__(self, **kwargs):
        self.id: int = kwargs['id']
        self.form: str = kwargs['form']
        self.label: str = kwargs['label']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']
        self.k_id: str = kwargs['kid'] if 'kid' in kwargs else kwargs['k_id']
        self.wiki_id: str = kwargs['wikiid'] if 'wikiid' in kwargs else kwargs['wiki_id']
        self.url: str = kwargs['URL'] if 'URL' in kwargs else kwargs['url']


class ELLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[ELItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[ELItem]:
        return self.data


class WSDItem(Item):
    def __init__(self, **kwargs):
        self.word: str = kwargs['word']
        self.sense_id: int = kwargs['sense_id']
        self.pos: str = kwargs['pos']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']
        self.word_id: int = kwargs['word_id']


class WSDLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[WSDItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[WSDItem]:
        return self.data

    def get_forms(self, index: int) -> list[tuple[str, str]]:
        return [(w.word, w.pos) for w in self.tolist() if w.word_id == index]


class DEPItem(Item):
    def __init__(self, **kwargs):
        self.word_id: int = kwargs['word_id']
        self.word_form: str = kwargs['word_form']
        self.head: int = kwargs['head']
        self.label: str = kwargs['label']
        self.dependent: list[int] = kwargs['dependent']


class DEPLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[DEPItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[DEPItem]:
        return self.data


class SRLPredicate(Item):
    def __init__(self, **kwargs):
        self.form: str = kwargs['form']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']
        self.lemma: str = kwargs['lemma']


class SRLArgument(Item):
    def __init__(self, **kwargs):
        self.form: str = kwargs['form']
        self.label: str = kwargs['label']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']
        self.word_id: int = kwargs['word_id']


class SRLItem(Item):
    def __init__(self, **kwargs):
        self.predicate: SRLPredicate = SRLPredicate(**kwargs['predicate'])
        self.argument: list[SRLArgument] = [SRLArgument(**arg) for arg in kwargs['argument']]

    def __repr__(self):
        return f'<{self.__class__.__name__}→\n{pprint.pformat(self.__dict__, indent=2)}>'


class SRLLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[SRLItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[SRLItem]:
        return self.data


class CRMention(Item):
    def __init__(self, **kwargs):
        self.sentence_id: str = kwargs['sentence_id']
        self.form: str = kwargs['form']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']
        self.ne_id: int = kwargs['NE_id'] if 'NE_id' in kwargs else kwargs['ne_id']


class CRItem(Item):
    def __init__(self, **kwargs):
        self.mention: list[CRMention] = [CRMention(**arg) for arg in kwargs['mention']]


class CRLayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[CRItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[list[CRMention]]:
        return [item.mention for item in self.data]


class ZAPredicate(Item):
    def __init__(self, **kwargs):
        self.form: str = kwargs['form']
        self.sentence_id: str = kwargs['sentence_id']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']


class ZAAntecedent(Item):
    def __init__(self, **kwargs):
        self.form: str = kwargs['form']
        self.type: str = kwargs['type']
        self.sentence_id: str = kwargs['sentence_id']
        self.begin: int = kwargs['begin']
        self.end: int = kwargs['end']


class ZAItem(Item):
    def __init__(self, **kwargs):
        self.predicate: ZAPredicate = ZAPredicate(**kwargs['predicate'])
        self.antecedent: list[ZAAntecedent] = [ZAAntecedent(**a_kwargs) for a_kwargs in kwargs['antecedent']]

    def __repr__(self):
        return f'<{self.__class__.__name__}→\n{pprint.pformat(self.__dict__, indent=2)}>'


class ZALayer(Layer):
    def __init__(self, layer: str, data: list, super_instance=None):
        super().__init__(layer=layer, data=[ZAItem(**d) for d in data], super_instance=super_instance)

    def tolist(self) -> list[ZAItem]:
        return self.data
