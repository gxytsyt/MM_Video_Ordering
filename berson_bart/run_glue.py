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


def init_model(args, TS, model_name: str, device, do_lower_case: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    if TS == "T":
        model = AutoModelForsentenceordering.from_pretrained(model_name)
    else:
        model = AutoModelForsentenceordering_student.from_pretrained(model_name, temp=args.temp, lbd=args.lambda_, momentum=args.momentum)

    model.to(device)
    model.eval()
    return tokenizer, model


def train(args, train_dataset, model, tokenizer, train_num=50):
    # train_num = 1
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=preprocess)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    lr_bart = args.lr_bart
    lr_contrast_cross = args.lr_contrast_cross
    lr_ordering = args.lr_ordering

    bart_related_para = ['model.', 'model_up2.']
    contrast_cross_attn_para = ['cls_token_video', 'pos_embed_video', 'cls_token_text', 'pos_embed_text', 'visn_fc.',
                                'video_encoder.', 'text_encoder.', '_proj_con.', 'mm_cross_decoder.', 'mm_final_ln.']
    ordering_related = ['key_linear.', 'query_linear.', 'tanh_linear.', 'encoder.transformer_inter.',
                        'encoder.layer_norm.', 'two_level_encoder.', 'pw_k.', 'tran_decoder.']

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in bart_related_para) and not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": lr_bart,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in bart_related_para) and any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr_bart,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in contrast_cross_attn_para) and not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": lr_contrast_cross,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in contrast_cross_attn_para) and any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr_contrast_cross,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in ordering_related) and not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": lr_ordering,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in ordering_related) and any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr_ordering,
        },
    ]

    optimizer = AdamW(
        # optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        optimizer_grouped_parameters, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))


    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(args, model, tokenizer, '-1')
    #     for key, value in results.items():
    #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

    best_t = 0
    count_es = 0
    epoch = -1
    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            if epoch > 0:
                alpha = args.alpha
            else:
                alpha = args.alpha * min(1, step / len(epoch_iterator))

            inputs = {
                       'pairs_list':         batch[0],
                       'passage_length':     batch[1],
                       "pairs_num":          batch[2],
                       # "sep_positions":      batch[6],
                       "ground_truth":       batch[3],
                       "mask_cls":           batch[4],
                       # "pairwise_labels":    batch[9],
                       # "sentence_input_id":  batch[11],
                       "sentence_attention_mask": batch[6],
                       "sentence_length":    batch[7],
                       "para_input_id":       batch[8],
                       "para_attention_mask": batch[9],
                       "max_sentence_length": batch[10],
                       "imgs":                batch[11],
                       "mm_mask":             batch[12],
                "alpha": alpha,
                       "cuda":            args.cuda_ip}

            torch.autograd.set_detect_anomaly(True)
            ce_loss = model(**inputs)

            loss = ce_loss

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                logging_steps = args.logging_steps

                if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/logging_steps, global_step)
                    logging_loss = tr_loss

                    ####### early stop #######
                    taus = results['taus']
                    if taus > best_t:
                        count_es = 0
                        best_t = taus
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        count_es += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        results = evaluate(args, model, tokenizer, str(epoch))
        print(results)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


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

            tru = batch[3].view(-1).tolist()  # true order
            true_num = batch[1].view(-1)
            tru = tru[:true_num]
            truth.append(tru)

            with torch.no_grad():

                if len(tru) == 1:
                    pred = tru
                else:
                    pred = beam_search_pointer(args, model,
                        pairs_list=batch[0], passage_length=batch[1], pairs_num=batch[2],
                        ground_truth=batch[3], mask_cls=batch[4], sentence_attention_mask=batch[6], sentence_length=batch[7],
                        para_input_id=batch[8], para_attention_mask=batch[9], max_sentence_length=batch[10],
                        imgs=batch[11], mm_mask=batch[12],
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

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.fea_data_dir, 'bart_nopadding_cached_{}_{}_{}_mask'.format(
        # 'val' if evaluate else 'train',
        'test' if evaluate else 'train',
        list(filter(None, args.teacher_name_or_path.split('/'))).pop(),
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
        # examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        # examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                cached_features_file,
                                                evaluate=evaluate,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=1,
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        print('features', len(features))

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            if not os.path.exists(args.fea_data_dir):
                os.makedirs(args.fea_data_dir)
            torch.save(features, cached_features_file)  # save examples not padded

    dataset = features
    # features.view()
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # input('time')

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--fea_data_dir", default=None, type=str, required=True,
                        help="features data")

    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--teacher_name_or_path", default=None, type=str, required=True)
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
    parser.add_argument("--temperature", default=2.0, type=float,
                        help="Distillation temperature. Only for distillation.")
    parser.add_argument("--alpha_mseloss", default=100.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_senvec_pair_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_cls_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_docmat_mseloss", default=100.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_clskl_loss", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_attkl_loss", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--sen_layer_num", default=9, type=int)

    parser.add_argument("--temp", default=0.1, type=float,)
    parser.add_argument("--lambda_", default=0.8, type=float,)
    parser.add_argument("--momentum", default=0.995, type=float,)
    parser.add_argument("--alpha", default=0.4, type=float,)

    parser.add_argument("--lr_bart", default=1e-5, type=float,)
    parser.add_argument("--lr_contrast_cross", default=5e-5, type=float,)
    parser.add_argument("--lr_ordering", default=1e-5, type=float,)


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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    TS = 'S'
    tokenizer, model = init_model(args,
        TS, args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # train_dataset = None
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)


if __name__ == "__main__":
    main()
