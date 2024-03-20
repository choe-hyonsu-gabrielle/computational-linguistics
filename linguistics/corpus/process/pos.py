from difflib import SequenceMatcher
from collections import namedtuple
from transformers import PreTrainedTokenizer, AutoTokenizer
from linguistics.corpus.process import Process
from linguistics.corpus.collection import Sentence
from linguistics.corpus.layer import POSLayer
from linguistics.utils import subgroups


Subword = namedtuple(typename='Subword', field_names=['id', 'subword', 'trimmed', 'word_id', 'token_id'])


class POSSubwordMorphemeAlignment:
    def __init__(self, pos_layer: POSLayer, subwords: list[Subword], partition_by_word_ids=True):
        self.pos_layer = pos_layer
        self.morphemes = subgroups(pos_layer, by='word_id', starts_from=1) if partition_by_word_ids else pos_layer
        self.subwords = subgroups(subwords, by='word_id', starts_from=0) if partition_by_word_ids else subwords
        assert len(self.pos_layer.super.words) == len(self.morphemes) == len(self.subwords)

    def align(self):
        sentence: Sentence = self.pos_layer.super
        print(sentence.canonical_form)
        for word_id in range(len(sentence.words)):
            tokens: list[str] = [e.trimmed for e in self.subwords[word_id]]
            morphs: list[str] = [e.form for e in self.morphemes[word_id]]
            matcher = SequenceMatcher(isjunk=None, a=tokens, b=morphs)
            print(sentence.words[word_id])
            for opcode, t_begin, t_end, m_begin, m_end in matcher.get_opcodes():
                if opcode == 'equal':
                    labels = [e.label for e in self.morphemes[word_id][m_begin:m_end]]
                elif opcode == 'replace':
                    if t_end - t_begin == m_end - m_begin:
                        labels = [f'{e.form}/{e.label}' for e in self.morphemes[word_id][m_begin:m_end]]
                    elif 1 == t_end - t_begin < m_end - m_begin:
                        labels = ['+'.join([f'{e.form}/{e.label}' for e in self.morphemes[word_id][m_begin:m_end]])]
                    elif 1 == m_end - m_begin < t_end - t_begin:
                        labels = ['I-' + self.morphemes[word_id][m_begin:m_end][0].label] * (t_end - t_begin)
                        labels[0] = 'B' + labels[0][1:]
                        labels[-1] = 'E' + labels[-1][1:]
                    else:
                        labels = [f'?-{e.form}/{e.label}' for e in self.morphemes[word_id][m_begin:m_end]]
                elif opcode == 'insert':
                    labels = [f'X-{e.form}/{e.label}' for e in self.morphemes[word_id][m_begin:m_end]]
                elif opcode == 'delete':
                    labels = [f'X-{e.form}/{e.label}' for e in self.morphemes[word_id][m_begin:m_end]]
                else:
                    raise NotImplementedError(opcode)
                print(f'\t"{opcode:7}" t[{t_begin}:{t_end}] vs m[{m_begin}:{m_end}] - {tokens[t_begin:t_end]} → {labels}')
            print()


class POSProcess(Process):
    def __init__(self, tokenizer: PreTrainedTokenizer, contact_prefix: str = '##'):
        super().__init__()
        self.tokenizer = tokenizer
        self.contact_prefix = contact_prefix

    def __call__(self, pos_layer: POSLayer = None, *args, **kwargs):
        sentence: Sentence = pos_layer.super
        input_ids, offset_mapping = self.tokenizer.encode_plus(
            text=sentence.canonical_form,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=False
        ).values()
        if input_ids[0] in self.tokenizer.all_special_ids:
            input_ids = input_ids[1:]
            offset_mapping = offset_mapping[1:]
        if input_ids[-1] in self.tokenizer.all_special_ids:
            input_ids = input_ids[:-1]
            offset_mapping = offset_mapping[:-1]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        word_ids = self.offset_to_word_ids(offset_mapping)
        assert len(tokens) == len(word_ids)

        subwords = []
        for idx, (subword, word_id, token_id) in enumerate(zip(tokens, word_ids, input_ids)):
            trimmed = subword[len(self.contact_prefix):] if subword.startswith(self.contact_prefix) else subword
            subwords.append(Subword(idx, subword, trimmed, word_id, token_id))

        return POSSubwordMorphemeAlignment(
            pos_layer=pos_layer,
            subwords=subwords,
            partition_by_word_ids=True
        )

    @staticmethod
    def offset_to_word_ids(offset_mapping, strip=False):
        """
        returns word ids map from offset_mapping
        :param offset_mapping: from PretrainedTokenizer
        :param strip: if true, remove all [0, 0] elements outside valid subword tokens such as [CLS], [SEP], [PAD]
        :return: list of word_id, starts from 0
        """
        if strip:
            while offset_mapping[0] == [0, 0] or offset_mapping[-1] == [0, 0]:
                if offset_mapping[0] == [0, 0]:
                    offset_mapping = offset_mapping[1:]
                if offset_mapping[-1] == [0, 0]:
                    offset_mapping = offset_mapping[:-1]
        word_ids = []
        current_id = 0
        if len(offset_mapping) == 1:
            word_ids.append(current_id)
            return word_ids
        for (a_start, a_end), (b_start, b_end) in zip(offset_mapping, offset_mapping[1:]):
            if 0 == a_start < a_end == b_start < b_end:
                word_ids.append(current_id)
                word_ids.append(current_id)
            elif 0 == a_start < a_end < b_start < b_end:
                word_ids.append(current_id)
                current_id += 1
                word_ids.append(current_id)
            elif a_start < a_end == b_start < b_end:
                word_ids.append(current_id)
            elif a_start < a_end < b_start < b_end:
                current_id += 1
                word_ids.append(current_id)
        return word_ids


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    pos = POSProcess(tokenizer=tokenizer)
    pos()
