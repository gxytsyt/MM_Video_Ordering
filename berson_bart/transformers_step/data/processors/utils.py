# coding=utf-8
import csv
import random
import sys
import copy
import json
import os
import glob
from tqdm import tqdm

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    # def __init__(self, guid, text_a, text_b=None, label=None):
    def __init__(self, guid, text_seq, img_path_seq, label=None):
        self.guid = guid
        self.text_seq = text_seq
        self.img_path_seq = img_path_seq
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, position_ids, order, cls_ids, mask_cls, num_sen, span_index):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.order = order
        self.cls_ids = cls_ids
        self.mask_cls = mask_cls
        self.num_sen = num_sen
        self.span_index=span_index

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, data_dir, quotechar=None):
        json_file = data_dir + '/pure_video_data.json'

        story_seqs = []
        false_section = []
        max_story_length = 15

        video_fea_root = data_dir + '/clip_features'

        len_seq = []
        with open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)

            random.seed(42)
            random.shuffle(datas)

            len_test_set = int(len(datas) * 0.1)

            if quotechar == 'train':
                datas = datas[:-len_test_set]
            if quotechar == 'test':
                datas = datas[-len_test_set:]

            for item in tqdm(datas):
                task_name = item['task_name']
                section_name = item['section']['section_name']
                steps_all = item['section']['steps_all']

                section_id_my = str(task_name) + str(section_name)
                story_seq = [section_id_my]

                try:
                    for step_i, step in enumerate(steps_all):
                        video_url = step['video_url']
                        step_full_text = step['step_full_text']
                        video_feature_path = video_fea_root + str(video_url) + '.pkl'

                        assert os.path.isfile(video_feature_path)

                        element = (step_full_text, video_feature_path)
                        story_seq.append(element)

                    if len(story_seq) <= 2:
                        continue

                    story_seq = story_seq[:max_story_length + 1]
                    story_seqs.append(story_seq)
                except Exception as e:
                    print('false section_id:', section_id_my)
                    false_section.append(section_id_my)
                    print(e)


        print("There are {} valid story sequences".format(len(story_seqs)))  #
        print("false_section:", false_section)

        return story_seqs
