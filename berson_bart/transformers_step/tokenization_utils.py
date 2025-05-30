# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization classes for python tokenizers.
    For fast tokenizers (provided by HuggingFace's tokenizers library) see tokenization_utils_fast.py
"""

import itertools
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

from .file_utils import add_end_docstrings
from .tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

from random import shuffle
import itertools
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

import torch
import skimage
from torchvision import transforms, utils
from skimage import io, transform
from skimage.transform import resize as skimage_resize
from skimage import img_as_ubyte
import pickle

logger = logging.getLogger(__name__)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))

# Adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class PreTrainedTokenizer(PreTrainedTokenizerBase):
    """ Base class for all slow tokenizers.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

    - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
      required by the model, and as associated values, the filename for saving the associated file (string).
    - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
      being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
      `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
      associated pretrained vocabulary file.
    - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
      models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
      model has no maximum input size.
    - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
      pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
      ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
      ``from_pretrained()`` method.

    Args:
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``


    .. automethod:: __call__
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    def get_vocab(self):
        """ Returns the vocabulary as a dict of {token: index} pairs. `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the vocab. """
        raise NotImplementedError()

    def get_added_vocab(self) -> Dict[str, int]:
        return self.added_tokens_encoder

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens=False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string. Each string is a token to add. Tokens are only added if they are not
                already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and self.init_kwargs.get("do_lower_case", False):
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if special_tokens:
            self.unique_no_split_tokens = list(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            # Or on the newly added tokens
            self.unique_no_split_tokens = list(set(self.unique_no_split_tokens).union(set(tokens_to_add)))

        return len(tokens_to_add)

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: TextInput, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Args:
                text (:obj:`string`): The sequence to be encoded.
                **kwargs (:obj: `dict`): Arguments passed to the model-specific `prepare_for_tokenization` preprocessing method.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        )

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if self.init_kwargs.get("do_lower_case", False):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        # Try to avoid splitting on token
                        if (
                            i < len(split_text) - 1
                            and not _is_end_of_word(sub_text)
                            and not _is_start_of_word(split_text[i + 1])
                        ):
                            # Don't extract the special token
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result += [full_word]
                            full_word = ""
                            continue
                    # Strip white spaces on the right
                    if tok_extended.rstrip and i > 0:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        sub_text = sub_text.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_no_split_tokens else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """
        raise NotImplementedError


    def encode_plus_my(self,
                    text,
                    image,
                    text_pair=None,
                    add_special_tokens=False,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    return_tensors=None,
                    **kwargs):

        return self.prepare_for_model_my(text, image,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_tensors=return_tensors,
                                      **kwargs)

    def pairs_generator(self, lenth):
        '''Generate the combinations of sentence pairs

            Args:
                lenth (int): the length of the sentence
            
            Returns:
                combs (list) : all combination of the index pairs in the passage
                num_combs (int) : the total number of all combs
        '''
        indices = list(range(lenth))
        combs_one_side = list(itertools.combinations(indices, 2))
        combs_one_side = [[x1, x2] for x1,x2 in combs_one_side]
        combs_other_side = [[x2, x1] for x1,x2 in combs_one_side]
        combs = combs_one_side + combs_other_side
        return combs, len(combs)

    def read_and_transform_img_from_filename(self, filename, img_transform_func=None):
        def read_img_from_filename(filename):
            try:
                img = io.imread(filename, )
            except:
                img = io.imread(filename, plugin="imageio")
            return img

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_transform_func = transforms.Compose([
            Rescale((224, 224)),
            ToTensor(),
            normalize,
        ])
        img = read_img_from_filename(filename)
        # print(filename)
        # print('img.shape0', img.shape)
        if len(img.shape) < 3:  # Grayscale to RGB?
            img = skimage.color.gray2rgb(img)
        if img.shape[-1] > 3:  # Remove the alpha channel.
            img = img[:, :, :3]
        # print('img.shape1', img.shape)
        if img_transform_func is not None:
            img = img_transform_func(img)
        return img


    def prepare_for_model_my(self, ids, img, max_length=None, add_special_tokens=False, stride=0,
                          truncation_strategy='longest_first', return_tensors=None, **kwargs):

        sents = ids
        imgs = []
        for img_path in img:
            with open(img_path, 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
                img_my = pickle.load(file)

            imgs.append(img_my)

        max_len_new = 15
        imgs = imgs[:max_len_new]

        sents = sents[:max_len_new]
        passage_length = len(sents)

        # shuffle 
        base_index = list(range(passage_length)) 
        shuffled_index = base_index 
        shuffle(shuffled_index) 
        ground_truth = list(np.argsort(shuffled_index))
        pairs_list, pairs_num = self.pairs_generator(passage_length)

        input_ids = []
        masked_ids = []
        token_type_ids = []
        sep_positions = []
        pairwise_labels = []

        max_sample_len = 0 
        # max_length_new = 40
        max_length_new = 60

        sfdoc = []  # [1 2 4 3 0]
        sfimg = []
        for ri in shuffled_index:
            sentence = sents[ri]
            sentence0 = sentence.replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ').replace('\u00a0', ' ')
            sentence = re.sub(' +', ' ', sentence0)
            sfdoc.append(sentence)
            sfimg.append(imgs[ri])

        imgs = torch.stack(sfimg)

        sentence_input_id = []
        sentence_attention_mask = []
        sentence_length = []

        for sentence in sfdoc:

            tokenized_sent = self.tokenize(sentence, **kwargs)
            tokenized_sent = tokenized_sent[:max_length_new]

            tokenized_sent = ['<s>'] + tokenized_sent + ['</s>']
            input_id = self.convert_tokens_to_ids(tokenized_sent)
            masked_id = [1] * len(input_id)

            sentence_input_id.append(input_id)
            sentence_attention_mask.append(masked_id)
            sentence_length.append(len(input_id))

        # "sep": ["</s>", 2]
        # "cls": ["<s>", 0]
        para_input_id = [0]
        para_attention_mask = [1]
        para_length = 1

        sent_len_for_mask = []

        for sentence in sfdoc:
            tokenized_sent = self.tokenize(sentence, **kwargs)
            tokenized_sent = tokenized_sent[:max_length_new]

            tokenized_sent_para = tokenized_sent + ['</s>']
            input_id_para_sent = self.convert_tokens_to_ids(tokenized_sent_para)
            masked_id_parasent = [1] * len(input_id_para_sent)

            sent_len_for_mask.append(len(input_id_para_sent))

            para_input_id = para_input_id + input_id_para_sent
            para_attention_mask = para_attention_mask + masked_id_parasent
            para_length = para_length + len(input_id_para_sent)



        for sent_id1, sent_id2 in pairs_list:
            sep_position = [0,0]

            ### pairwise label ###
            first_index = shuffled_index[sent_id1]
            sec_index = shuffled_index[sent_id2]

            if first_index > sec_index:
                pairwise_label = 0

            if first_index < sec_index:
                pairwise_label = 1
            pairwise_labels.append(pairwise_label)


            sent1 = sents[shuffled_index[sent_id1]]
            sent2 = sents[shuffled_index[sent_id2]]

            tokenized_sent1 = self.tokenize(sent1, **kwargs)
            tokenized_sent1 = tokenized_sent1[:max_length_new]
            tokenized_sent2 = self.tokenize(sent2, **kwargs)
            tokenized_sent2 = tokenized_sent2[:max_length_new]

            concat_sents = ['<s>'] + tokenized_sent1 + ['</s>']
            sep_position[0] = len(concat_sents) - 1
            token_type_id = [0] * len(concat_sents)
            concat_sents += tokenized_sent2 + ['</s>']
            input_id = self.convert_tokens_to_ids(concat_sents)


            sep_position[1] = len(concat_sents) - 1
            token_type_id += [1] * len(tokenized_sent2 + ['</s>'])
            masked_id = [1] * len(token_type_id)

            max_sample_len = len(masked_id) if max_sample_len < len(masked_id) else max_sample_len

            input_ids.append(input_id)
            masked_ids.append(masked_id)
            token_type_ids.append(token_type_id)
            sep_positions.append(sep_position)
        


        encoded_inputs = {}
        encoded_inputs['input_ids'] = input_ids
        encoded_inputs['masked_ids'] = masked_ids
        encoded_inputs['token_type_ids'] = token_type_ids
        encoded_inputs['sep_positions'] = sep_positions
        encoded_inputs['shuffled_index'] = shuffled_index
        encoded_inputs['max_sample_len'] = max_sample_len
        encoded_inputs['ground_truth'] = ground_truth
        encoded_inputs['passage_length'] = passage_length
        encoded_inputs['pairs_num'] = pairs_num 
        encoded_inputs['pairs_list'] = pairs_list
        encoded_inputs['pairwise_labels'] = pairwise_labels


        encoded_inputs['sentence_input_id'] = sentence_input_id
        encoded_inputs['sentence_attention_mask'] = sentence_attention_mask
        encoded_inputs['sentence_length'] = sentence_length


        encoded_inputs['para_input_id'] = para_input_id
        encoded_inputs['para_attention_mask'] = para_attention_mask
        encoded_inputs['para_length'] = para_length

        # img
        encoded_inputs['imgs'] = imgs

        # mask
        encoded_inputs['sent_len_for_mask'] = sent_len_for_mask

        if passage_length > 15 or len(para_input_id)==0:
            return None
        else:
            return encoded_inputs



    def convert_tokens_to_ids(self, tokens):
        """ Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError


    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_pretokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_pretokenized:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_pretokenized=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_pretokenized:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_pretokenized=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self._prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_pretokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs) 
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_pretokenized:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_pretokenized=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_pretokenized and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self._prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding_strategy=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        prepend_batch_axis: bool = False,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length and verbose:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.model_max_length)
            )

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def prepare_for_tokenization(self, text: str, is_pretokenized=False, **kwargs) -> (str, dict):
        """ Performs any necessary transformations before tokenization.

            This method should pop the arguments from kwargs and return kwargs as well.
            We test kwargs at the end of the encoding process to be sure all the arguments have been used.
        """
        return (text, kwargs)

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "only_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """ Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy (:obj:`string`, `optional`, defaults to "only_first"):
                String selected in the following options:

                - 'only_first' (default): Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'longest_first': Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences).
                  Overflowing tokens only contains overflow from the first sequence.
                - 'do_not_truncate'
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input"
                    f"but the first sequence has a length {len(ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_second'."
                )
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                overflowing_tokens = pair_ids[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input"
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. This implementation does not add special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[int, List[int]]:
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
        return " ".join(self.convert_ids_to_tokens(tokens))

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory) -> Tuple[str]:
        """ Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
            and special token mappings.

            Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the full
            Tokenizer state if you want to reload it using the :func:`~transformers.PreTrainedTokenizer.from_pretrained`
            class method.
        """
        raise NotImplementedError
