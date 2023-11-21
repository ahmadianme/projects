import os

import torch
import torch.nn as nn

from arch.tokenization import BertTokenizer
from arch.modeling import ARCHFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG




from arguments import args
from arch.modeling import ARCHModel, BertPreTrainingHeads, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

useGPU = torch.cuda.is_available()
device = torch.device('cuda') if useGPU else torch.device('cpu')

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


# def convert_sents_to_features(sents, max_seq_length, tokenizer):
#     """Loads a data file into a list of `InputBatch`s."""
#
#     features = []
#     for (i, sent) in enumerate(sents):
#         tokens_a = tokenizer.tokenize(sent.strip())
#
#         # Account for [CLS] and [SEP] with "- 2"
#         if len(tokens_a) > max_seq_length - 2:
#             tokens_a = tokens_a[:(max_seq_length - 2)]
#
#         # Keep segment id which allows loading BERT-weights.
#         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#         segment_ids = [0] * len(tokens)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_mask=input_mask,
#                               segment_ids=segment_ids))
#     return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers





class XTECModelBase(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr'):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.config = config
        self.bert = ARCHModel(config)
        self.mode = mode

        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_bert_weights)



    def calculateLoss(self, lang_output, pooled_output, masked_lm_labels, mode='predict'):

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)

        masked_lm_loss = None

        if mode == 'train':
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )

        return lang_prediction_scores, cross_relationship_score, masked_lm_loss




    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None):

        return self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask)








class XTECModel(nn.Module):
    def __init__(self,):
        super().__init__()

        set_visual_config(args)

        self.bert = XTECModelBase.from_pretrained(
            "bert-base-uncased"
        )


        # # Weight initialization
        # self.apply(self.init_bert_weights)





    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, mode='predict'):

        (lang_output, visn_output), pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=(visual_feats, pos),
        )

        lang_prediction_scores, cross_relationship_score, loss = self.bert.calculateLoss(lang_output, pooled_output, masked_lm_labels, mode)

        if mode == 'train':
            return lang_prediction_scores, cross_relationship_score, loss

        return lang_prediction_scores, cross_relationship_score







    def load(self, path):
        # Load state_dict from snapshot file
        print("Load pre-trained model from %s" % path)
        if useGPU:
            state_dict = torch.load("%s_ARCH.pth" % path)
        else:
            state_dict = torch.load("%s_ARCH.pth" % path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.bert.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.bert.load_state_dict(state_dict, strict=False)


