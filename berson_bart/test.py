
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange


from transformers_step import glue_compute_metrics as compute_metrics
from transformers_step import glue_output_modes as output_modes
from transformers_step import glue_processors as processors
from transformers_step import glue_convert_examples_to_features as convert_examples_to_features

from preprocess_batch import preprocess
import torch.nn as nn

from transformers_step import AutoModelForsentenceordering, AutoTokenizer
from transformers_step import AutoModelForsentenceordering_student
from transformers_step import AdamW, get_linear_schedule_with_warmup
from transformers_step.modeling_bart_student import beam_search_pointer

import torch.nn.functional as F

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_model(TS, model_name: str, device, do_lower_case: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    if TS == "T":
        model = AutoModelForsentenceordering.from_pretrained(model_name)
    else:
        model = AutoModelForsentenceordering_student.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return tokenizer, model


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=preprocess)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        f = open(os.path.join(args.output_dir, "output_order.txt"), 'w')

        best_acc = []
        truth = []
        predicted = []
      
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            tru = batch[7].view(-1).tolist()  # true order
            true_num = batch[4].view(-1)
            tru = tru[:true_num]
            truth.append(tru)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                           'attention_mask': batch[1],
                           'token_type_ids': batch[2],
                           'pairs_list':    batch[3],
                           'passage_length':          batch[4],
                           "pairs_num":         batch[5],
                           "sep_positions":          batch[6],
                           "ground_truth":          batch[7],
                           "mask_cls":        batch[8],
                           "pairwise_labels":    batch[9],
                           "cuda":            args.cuda_ip}

                if len(tru) == 1:
                    pred = tru
                else:
                    pred = beam_search_pointer(args, model, input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                        pairs_list=batch[3], passage_length=batch[4], pairs_num=batch[5], sep_positions=batch[6], 
                        ground_truth=batch[7], mask_cls=batch[8], pairwise_labels=batch[9],
                        sentence_input_id=batch[11], sentence_attention_mask=batch[12], sentence_length=batch[13],
                        para_input_id=batch[14], para_attention_mask=batch[15], max_sentence_length=batch[16],
                                               imgs=batch[17], sent_len_for_mask=batch[18], mm_mask=batch[19],
                                               cuda=args.cuda_ip)

                predicted.append(pred)
                print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                      file=f)                

        right, total = 0, 0
        pmr_right = 0
        taus = []
        accs = []
        pm_p, pm_r = [], []
        import itertools

        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
            if len(p) == 1:
                right += 1
                total += 1
                pmr_right += 1
                taus.append(1)
                continue

            eq = np.equal(t, p)
            right += eq.sum()
            accs.append(eq.sum()/len(t))

            total += len(t)

            pmr_right += eq.all()

            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2

            taus.append(tau)

        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                             list(itertools.chain.from_iterable(predicted)))

        best_acc.append(acc)

        pmr = pmr_right / len(truth)

        taus = np.mean(taus)

        pm_p = np.mean(pm_p)
        pm_r = np.mean(pm_r)
        pm = 2 * pm_p * pm_r / (pm_p + pm_r)

        f.close()
        accs = np.mean(accs)

        results['acc'] = accs
        results['pmr'] = pmr
        results['taus'] = taus
        results['pm'] = pm

        output_only_eval_file_1 = os.path.join(args.output_dir, "all_test_results.txt")
        fh = open(output_only_eval_file_1, 'a')
        fh.write(prefix)
        for key in sorted(results.keys()):
            fh.write("%s = %s\n" % (key, str(results[key])))
        fh.close()

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.fea_data_dir, 'bart_nopadding_cached_{}_{}_{}_mask'.format(
        'test' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                evaluate=evaluate,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=1,
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        print ('features', len(features))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)  # save examples not padded

    dataset = features  

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache


    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--fea_data_dir", default=None, type=str, required=True)

    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--cuda_ip", default="cuda:0", type=str,
                        help="Total number of training epochs to perform.")


    #### paragraph encoder ####
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int) 
    parser.add_argument("--para_dropout", default=0.1, type=float,
                        help="Total number of training epochs to perform.")

    #### pointer ###
    parser.add_argument("--beam_size", default=64, type=int)

    #### pairwise loss ###
    parser.add_argument("--pairwise_loss_lam", default=0.1, type=float,help="Total number of training epochs to perform.")

    #### transformer decoder ###
    parser.add_argument("--decoder_layer", default=2, type=int) 
    parser.add_argument("--dec_heads", default=8, type=int)

    #### Distillation ###
    parser.add_argument("--temperature", default=10.0, type=float,
                        help="Distillation temperature. Only for distillation.")
    parser.add_argument("--alpha_mseloss", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_senvec_pair_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_cls_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_docmat_mseloss", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_clskl_loss", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_attkl_loss", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")


    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(args.cuda_ip if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    tokenizer, model = init_model('S',
        args.output_dir, device=args.device, do_lower_case=args.do_lower_case)

    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
