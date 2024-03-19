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

    def __repr__(self):
        sentence: Sentence = self.pos_layer.super
        words = sentence.words
        length = len(words)
        ref_id = sentence.ref_id
        return (f'[{self.__class__.__name__}] ref_id: "{ref_id}", len: {length}, sentence: "{sentence}", words: {words}\n'
                + '\n'.join([f'\t{i:2d} sub: {self._get_trimmed_sub(i)}, morph: {self._get_morphemes(i)}' for i in range(length)]))

    def align(self):
        sentence: Sentence = self.pos_layer.super
        for word_id in range(len(sentence.words)):
            op_code, mapping = self.word_level_alignment(word_id)

    def _get_trimmed_sub(self, i) -> list[str]:
        return [s.trimmed for s in self.subwords[i]]

    def _get_morphemes(self, i) -> list[str]:
        return [s.morpheme for s in self.morphemes[i]]

    def word_level_alignment(self, word_id: int):
        """
        This is a function which has dynamic programming approach partially, returns
        :param word_id: word_id to align
        :return:
        """
        subwords = self._get_trimmed_sub(word_id)
        morphemes = self._get_morphemes(word_id)

        if subwords == morphemes:
            # 완전 일치. ex) sub: ['역할', '을'], morph: ['역할', '을'] -> 그냥 zip()해서 return 하면 끝
            op_code = 'EXACT-ALIGNED'
            mapping = list(zip(self.subwords[word_id], self.morphemes[word_id]))
            return op_code, mapping
        elif len(subwords) == 1:
            # subword 길이가 1. ex) sub: ['찾아보기'], morph: ['찾아보', '기'] -> 그냥 1:n으로 대응시킴
            #                      sub: ['어떻게'], morph: ['어떻', '게']
            op_code = 'ONE-TO-MANY'
            mapping = (self.subwords[word_id][0], self.morphemes[word_id])
            return op_code, mapping
        elif len(subwords) > 1 and len(morphemes) == 1:
            # subword 길이가 1. ex) sub: ['우리', '나라'], morph: ['우리나라'] -> 길이 상관 없이 POS 태그를 B, I로 분할하여 대응
            # sub: ['그', '~'], morph: ['그'] 이런 경우처럼 오류가 있을 수도 있음
            op_code = 'MANY-TO-ONE'
        elif len(subwords) > 1 and len(morphemes) > 1:
            # sub > morph: ex) sub: ['아니', '겠', '습', '니까', '?'], morph: ['아니', '겠', '습니까', '?']
            #                  sub: ['말', '이', '에', '요', '.'], morph: ['말', '이', '에요', '.']
            #                  sub: ['학교', '에', '서'], morph: ['학교', '에서']
            #                  sub: ['규탄', '대', '회', '를'], morph: ['규탄', '대회', '를']
            #                  sub: ['부러워', '하', '더라', '니까'], morph: ['부러워하', '더라니까']
            #                  sub: ['취업', '난', '으로'], morph: ['취업난', '으로']
            #                  sub: ['핵', '미사', '일로'], morph: ['핵미사일', '로']
            #                  sub: ['안전', '보', '장이', '사회'], morph: ['안전', '보장', '이사회']

            # sub = morph: ex) sub: ['나가', '서'], morph: ['나가', '아서']
            #                  sub: ['폐', '장', '했', '잖아'], morph: ['폐장', '하', '았', '잖아']
            #                  sub: ['나라', '들이', '었', '거', '든', '요'], morph: ['나라', '들', '이', '었', '거든', '요']
            #                  sub: ['놉', '시다', '.'], morph: ['놓', 'ㅂ시다', '.']
            #                  sub: ['양성', '할', '게요', '"', ';', '충북', '대'], morph: ['양성', '하', 'ㄹ게', '요', '"', ';', '충북대']
            #                  sub: ['말', '인', '데'], morph: ['말', '이', 'ㄴ데']
            #                  sub: ['굉장', '한'], morph: ['굉장하', 'ㄴ']

            # sub < morph: ex) sub: ['맞아', '주', '고'], morph: ['맞', '아', '주', '고']
            #                  sub: ['반', '데', '.'], morph: ['바', '이', 'ㄴ데', '.']
            #                  sub: ['생겼', '다', '?'], morph: ['생기', '었', '다', '?']
            #                  sub: ['치명', '적', '인'], morph: ['치명', '적', '이', 'ㄴ']
            #                  sub: ['떴', '잖아', '.'], morph: ['뜨', '었', '잖아', '.']
            #                  sub: ['그래', '갖', '고'], morph: ['그러', '어', '갖', '고']
            #                  sub: ['드세요', '.'], morph: ['들', '시', '어요', '.']
            #                  sub: ['나아질', '까요', '?'], morph: ['나아지', 'ㄹ까', '요', '?']
            #                  sub: ['연구', '하시', '는'], morph: ['연구', '하', '시', '는']
            #                  sub: ['넣', '어', '주', '시고'], morph: ['넣', '어', '주', '시', '고']
            #                  sub: ['무력', '화', '할'], morph: ['무력', '화', '하', 'ㄹ']
            op_code = 'MANY-TO-MANY'
        else:
            raise NotImplemented


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
