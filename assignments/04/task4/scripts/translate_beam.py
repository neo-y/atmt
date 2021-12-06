import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='assignments/03/prepared', help='path to data directory')
    parser.add_argument('--dicts', required=True, help='path to directory containing source and target dictionaries')
    parser.add_argument('--checkpoint-path', default='checkpoints_asg4/checkpoint_best.pt',
                        help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')

    # Add beam search arguments
    parser.add_argument('--beam-size', default=5, type=int, help='number of hypotheses expanded in beam search')
    # alpha hyperparameter for length normalization (described as lp in https://arxiv.org/pdf/1609.08144.pdf equation 14)
    parser.add_argument('--alpha', default=0.0, type=float, help='alpha for softer length normalization')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.dicts, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.dicts, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                              batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                         args.batch_size, 1, 0, shuffle=False,
                                                                         seed=args.seed))
    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)

    # Iterate over the test set
    all_hyps_1  = {}
    all_hyps_2 = {}
    all_hyps_3 = {}
    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch
        batch_size = sample['src_tokens'].shape[0]
        searches = [BeamSearch(args.beam_size, args.max_len - 1, tgt_dict.unk_idx) for i in range(batch_size)]

        with torch.no_grad():
            # Compute the encoder output
            encoder_out = model.encoder(sample['src_tokens'], sample['src_lengths'])
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])

            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            # import pdb;pdb.set_trace()

            # Compute the decoder output at the first time step
            decoder_out, _ = model.decoder(go_slice, encoder_out)

            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size + 1, dim=-1)

        # Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j + 1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j + 1]
                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:, i, :]
                lstm_out = encoder_out['src_out'][0][:, i, :]
                final_hidden = encoder_out['src_out'][1][:, i, :]
                final_cell = encoder_out['src_out'][2][:, i, :]
                try:
                    mask = encoder_out['src_mask'][i, :]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, torch.cat((go_slice[i], next_word)), log_p, 1)
                # __QUESTION 3: Why do we add the node with a negative score?

                searches[i].add(-node.eval(args.alpha), node)

        # import pdb;pdb.set_trace()
        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len - 1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]
            if nodes == []:
                break  # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size + 1,
                                                    dim=-1)

            # Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):

                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j + 1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j + 1]
                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different?
                    # What would happen if we did not make this distinction?

                    # Store the node as final if EOS is generated
                    if next_word[-1] == tgt_dict.eos_idx:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                   next_word)), node.logp, node.length
                        )
                        search.add_final(-node.eval(args.alpha), node)

                    # Add the node to current nodes for next iteration
                    else:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                   next_word)), node.logp + log_p, node.length + 1
                        )
                        search.add(-node.eval(args.alpha), node)

            # #import pdb;pdb.set_trace()
            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            for search in searches:
                search.prune()

        # Segment into sentences
        best_sents_1 = []
        best_sents_2 = []
        best_sents_3 = []

        for search in searches:
            best = search.get_best()
            best_sents_1.append(best[0][1].sequence[1:].cpu())
            best_sents_2.append(best[1][1].sequence[1:].cpu())
            best_sents_3.append(best[2][1].sequence[1:].cpu())

        best_sents_1 = torch.stack(best_sents_1, dim=0)
        best_sents_2 = torch.stack(best_sents_2, dim=0)
        best_sents_3 = torch.stack(best_sents_3, dim=0)

        #best_sents_1 = torch.stack([search.get_best()[0][1].sequence[1:].cpu() for search in searches])
        #best_sents_2 = torch.stack([search.get_best()[1][1].sequence[1:].cpu() for search in searches])
        #best_sents_3 = torch.stack([search.get_best()[2][1].sequence[1:].cpu() for search in searches])

        decoded_batch_1 = best_sents_1.numpy()
        decoded_batch_2 = best_sents_2.numpy()
        decoded_batch_3 = best_sents_3.numpy()

        # import pdb;pdb.set_trace()

        output_sentences_1 = [decoded_batch_1[row, :] for row in range(decoded_batch_1.shape[0])]
        output_sentences_2 = [decoded_batch_2[row, :] for row in range(decoded_batch_2.shape[0])]
        output_sentences_3 = [decoded_batch_3[row, :] for row in range(decoded_batch_3.shape[0])]

        output_sentences_all = [output_sentences_1, output_sentences_2, output_sentences_3]

        # __QUESTION 6: What is the purpose of this for loop?

        # 1
        temp_1 = list()
        for sent in output_sentences_1:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0:
                temp_1.append(sent[:first_eos[0]])
            else:
                temp_1.append(sent)
        output_sentences_1 = temp_1

        # Convert arrays of indices into strings of words
        output_sentences_1 = [tgt_dict.string(sent) for sent in output_sentences_1]

        for ii, sent in enumerate(output_sentences_1):
            all_hyps_1[int(sample['id'].data[ii])] = sent

        # 2
        temp_2 = list()
        for sent in output_sentences_2:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0:
                temp_2.append(sent[:first_eos[0]])
            else:
                temp_2.append(sent)
        output_sentences_2 = temp_2

        # Convert arrays of indices into strings of words
        output_sentences_2 = [tgt_dict.string(sent) for sent in output_sentences_2]

        for ii, sent in enumerate(output_sentences_2):
            all_hyps_2[int(sample['id'].data[ii])] = sent

        # 3
        temp_3 = list()
        for sent in output_sentences_3:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0:
                temp_3.append(sent[:first_eos[0]])
            else:
                temp_3.append(sent)
        output_sentences_3 = temp_3

        # Convert arrays of indices into strings of words
        output_sentences_3 = [tgt_dict.string(sent) for sent in output_sentences_3]

        for ii, sent in enumerate(output_sentences_3):
            all_hyps_3[int(sample['id'].data[ii])] = sent


    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps_1.keys())):
                out_file.write(all_hyps_1[sent_id] + '\n' + all_hyps_2[sent_id] + '\n' + all_hyps_3[sent_id] + '\n' )


if __name__ == '__main__':
    args = get_args()
    main(args)