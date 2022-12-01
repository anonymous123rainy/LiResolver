"""Experiment-running framework."""
import argparse
import importlib
from logging import debug

import numpy as np
#from pytorch_lightning.trainer import training_tricks
import torch
import torch.utils.data as Data
import pytorch_lightning as pl
#import lit_models
import yaml
import time
#from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel
#from pytorch_lightning.plugins import DDPPlugin
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ###

# from . import data
# from . import models
# from . import lit_models


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(name=module_name, package='.')
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False) # False
    parser.add_argument("--litmodel_class", type=str, default="BertLitModel") # TransformerLitModel
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="WIKI80") # DIALOGUE
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="RobertaForPrompt") # bert.BertForSequenceClassification
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--ossl2_label_type", type=str, default="relation") # relation or tail

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()

    # data_class = _import_class(f"data.{temp_args.data_class}") ###
    # model_class = _import_class(f"models.{temp_args.model_class}")
    # litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")
    # data_class = _import_class(f"{temp_args.data_class}")  ###
    # model_class = _import_class(f"{temp_args.model_class}")
    # litmodel_class = _import_class(f"{temp_args.litmodel_class}")
    # import data
    from .data.dialogue import WIKI80 as data_class
    from .models import RobertaForPrompt as model_class
    from .lit_models.transformer import BertLitModel as litmodel_class

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser




def load_re_model():
    parser = _setup_parser()
    args = parser.parse_args()

    print('args:', args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # model_class = _import_class(f"models.{args.model_class}")
    # litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
    # data_class = _import_class(f"data.{args.data_class}")
    from .data.dialogue import WIKI80 as data_class
    from .models import RobertaForPrompt as model_class
    from .lit_models.transformer import BertLitModel as litmodel_class

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    data = data_class(args, model)
    data_config = data.get_data_config()

    model.resize_token_embeddings(len(data.tokenizer))

    data.setup_1()

    ### 搭建lit_model
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')

    ### 【读取已经训练好的model】
    device = torch.device("cuda")

    print('torch.cuda.is_available(): ', torch.cuda.is_available())  # torch.cuda.is_available():  True

    best_model_path = os.path.dirname(os.path.abspath(__file__))+'/'+r'output/epoch=1-Eval/f1=0.97.ckpt'
    # lit_model.load_state_dict(torch.load(best_model_path, map_location="cuda:0")["state_dict"])
    # lit_model.to(device) ###
    lit_model.load_state_dict(torch.load(best_model_path)["state_dict"])

    print(next(lit_model.parameters()).device)

    return args, lit_model



def predict_re(args, lit_model):

    # model_class = _import_class(f"models.{args.model_class}")
    # data_class = _import_class(f"data.{args.data_class}")
    from .data.dialogue import WIKI80 as data_class
    from .models import RobertaForPrompt as model_class

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    data = data_class(args, model)
    data_config = data.get_data_config()

    # 读取测试数据 放进来
    data.setup_2()



    TPLG = []
    PRED = []

    ''' 预测 '''
    model.eval()
    with torch.no_grad():

        loader = data.test_dataloader()


        for A, B, C, _ in loader:

            # for ins in batch:
            #     print(ins)
            #     print(ins[0])
            #     print(ins[1])
            #     print(ins[2])

            input_ids = torch.cat([torch.unsqueeze(ins, dim=0) for ins in A])
            attention_mask = torch.cat([torch.unsqueeze(ins, dim=0) for ins in B])
            labels = torch.cat([torch.tensor(np.expand_dims(ins, 0)) for ins in C])
            # print('input_ids', input_ids.size())  # torch.Size([10, 256])
            # print('attention_mask', attention_mask.size())  # torch.Size([10, 256])
            # print('labels', labels.size())  # torch.Size([10])

            # 【预测】
            logits = lit_model.model(input_ids, attention_mask, return_dict=True).logits  #### torch.Size([10, 256, 50295])
            logits = lit_model.pvp(logits, input_ids)  # 在各个类别上的概率 # torch.Size([10, 19])
            test_pre_logits = logits.detach().cpu().numpy()  # (10, 19)
            preds = np.argmax(test_pre_logits, axis=-1)  # 预测的标签 # (10,) [ 4 12 13 13 13 13 14 13 13 13]
            # test_labels = labels.detach().cpu().numpy() # 实际的标签 # (10,) [7 7 7 7 7 7 7 7 7 7]
            # print('test_pre_logits: ', test_pre_logits.shape)
            # print('preds: ', preds.shape, preds)

            TPLG.append(test_pre_logits)
            PRED.append(preds)

    TPLG = np.concatenate(TPLG, axis=0)
    PRED = np.concatenate(PRED, axis=0)

    # return test_pre_logits, preds
    # print('TPLG: ', TPLG.shape)
    # print('PRED: ', PRED.shape)

    return TPLG, PRED




# re_args, re_model = load_re_model()
# test_pre_logits, preds = predict_re(args=re_args, lit_model=re_model)
# print(test_pre_logits)
# print(preds)
# import json
# with open('./re_test_preds.json', 'w', encoding="utf-8") as fw:
#     json.dump(preds, fw)
