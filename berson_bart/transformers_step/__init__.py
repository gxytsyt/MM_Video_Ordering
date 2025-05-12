__version__ = "2.1.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

from .data import (is_sklearn_available,
                   InputExample, InputFeatures, DataProcessor,
                   glue_output_modes, glue_convert_examples_to_features,
                   glue_processors, glue_tasks_num_labels)

if is_sklearn_available():
    from .data import glue_compute_metrics

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .tokenization_bart import BartTokenizer, BartTokenizerFast
from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_utils_base import (
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_auto import AutoConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

from .encoder import TransformerInterEncoder

# Modeling
if is_torch_available():
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward
    from .modeling_auto import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForsentenceordering,
        AutoModelForsentenceordering_student,
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_SENTENCE_ORDERING_MAPPING,
    )

    from .modeling_bart import (
        PretrainedBartModel,
        BartForSequenceClassification,
        BartModel,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
        BartForOrdering,
        beam_search_pointer,
    )

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )


