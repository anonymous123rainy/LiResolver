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


from tqdm import tqdm


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    print('args:', args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    data_config = data.get_data_config()
    model.resize_token_embeddings(len(data.tokenizer))

    #print('【data_test:', type(data.data_test))

    # gpt no config?

    # if "gpt" in args.model_name_or_path or "roberta" in args.model_name_or_path:
    #     tokenizer = data.get_tokenizer()
    #     model.resize_token_embeddings(len(tokenizer))
    #     model.update_word_idx(len(tokenizer))
    #     if "Use" in args.model_class:
    #         continous_prompt = [a[0] for a in tokenizer([f"[T{i}]" for i in range(1,3)], add_special_tokens=False)['input_ids']]
    #         continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, data.num_labels+1)], add_special_tokens=False)['input_ids']]
    #         discrete_prompt = [a[0] for a in tokenizer(['It', 'was'], add_special_tokens=False)['input_ids']]
    #         dataset_name = args.data_dir.split("/")[1]
    #         model.init_unused_weights(continous_prompt, continous_label_word, discrete_prompt, label_path=f"{args.model_name_or_path}_{dataset_name}.pt")
    data.setup()
    #relation_embedding = _get_relation_embedding(data)
    
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')


    '''
    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))
    
    # init callbacks
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=20,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None


    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
    )

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate
    
    print('【data:', type(data))
    print(data)

    trainer.fit(lit_model, datamodule=data) ######




    # two steps

    path = model_checkpoint.best_model_path
    print(f"best model save path {path}")

    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
    config = vars(args)
    config["path"] = path
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))
    '''

    ### 【读取已经训练好的model】

    print('torch.cuda.is_available(): ', torch.cuda.is_available()) # torch.cuda.is_available():  True

    best_model_path = r'./output/epoch=0-Eval/f1=0.96-v3.ckpt'
    # lit_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu'))["state_dict"])
    lit_model.load_state_dict(torch.load(best_model_path)["state_dict"])


    '''
    print("【args.two_steps = ", args.two_steps)
    if not args.two_steps: 
        test_output = trainer.test()
        print('【test_output:', test_output)

        step2_model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Step2Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True)
    '''



    '''
    #predict_dataloader = pl.LightningDataModule.from_datasets(None, None, None, data.data_test).predict_dataloader() 
    #predictions = trainer.predict(dataloaders=predict_dataloader)

    #predictions = trainer.predict(datamodule=data)
    

    print('【data_test:', type(data.data_test), len(data.data_test),  len(data.data_test[5]), data.data_test[5][0].size(), data.data_test[5][1].size(), data.data_test[5][2].size(), data.data_test[5][3].size(), ) # tensor的尺寸用.size()
    from torch.utils.data import DataLoader
    predict_dataloader = DataLoader([ins[0] for ins in data.data_test])  ### (先看10个测试样本)
    print('【predict_dataloader:', type(predict_dataloader))
    numm = 0
    for dt in predict_dataloader:
        numm += 1
        if numm > 3:
            break
        #print(type(dt), len(dt), dt)

    predictions = trainer.predict(dataloaders=predict_dataloader) ###
    
    print(len(predictions)) ## yes. the size of batches, 10
    print(predictions[0][0].detach().cpu().numpy().shape) # () (1, 256, 50295)

    # logits = np.concatenate([batch_outputs[0].detach().cpu().numpy() for batch_outputs in predictions]) #(10, 256, 50295)
    logits = torch.cat([batch_outputs[0] for batch_outputs in predictions], -1) # （-1和0都拼接的不太对，，，）
    print(type(logits), logits.shape) #

    labels = [ins[2] for ins in data.data_test]
    input_ids = torch.cat([ins[0] for ins in data.data_test], -1)
    print(type(input_ids), input_ids.shape) # (10, 256)

    preds = lit_model.pvp(logits, input_ids)

    print(preds.shape)
    print(preds)

    #print(predictions)
    '''

    # input_ids, attention_mask, labels, so = data.data_test
    input_ids = torch.cat([torch.unsqueeze(ins[0], dim=0) for ins in data.data_test])
    attention_mask = torch.cat([torch.unsqueeze(ins[1], dim=0) for ins in data.data_test])
    labels = torch.cat([torch.tensor(np.expand_dims(ins[2], 0)) for ins in data.data_test])
    print('input_ids', input_ids.size()) # torch.Size([10, 256])
    print('attention_mask', attention_mask.size()) # torch.Size([10, 256])
    print('labels', labels.size()) # torch.Size([10])

    #### 【用lit_model进行预测】
    logits = lit_model.model(input_ids, attention_mask, return_dict=True).logits #### 【【【
    print('logits', logits.size()) # torch.Size([10, 256, 50295])

    logits = lit_model.pvp(logits, input_ids)  # 每个样本的预测类别
    print('logits', logits.size()) # torch.Size([10, 19])

    test_logits = logits.detach().cpu().numpy()
    print('test_logits', test_logits.shape) # (10, 19)
    test_labels = labels.detach().cpu().numpy()
    print('【【【test_labels', test_labels.shape, test_labels) # (10,) [7 7 7 7 7 7 7 7 7 7]

    preds = np.argmax(test_logits, axis=-1)
    print('【【【preds', preds.shape, preds) # (10,) [ 4 12 13 13 13 13 14 13 13 13]




    '''
    if args.two_steps:
        # we build another trainer and model for the second training
        # use the Step2Eval/f1 

        # lit_model_second = TransformerLitModelTwoSteps(args=args, model=lit_model.model, data_config=data_config)
        step_early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=6, check_on_train_epoch_end=False)
        callbacks = [step_early_callback, step2_model_checkpoint]
        trainer_2 = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        )
        trainer_2.fit(lit_model, datamodule=data)
        trainer_2.test()
        # result = trainer_2.test(lit_model, datamodule=data)[0]
        # with open("result.txt", "a") as file:
        #     a = result["Step2Test/f1"]
        #     file.write(f"test f1 score: {a}\n")
        #     file.write(config_file_name + '\n')

    # trainer.test(datamodule=data)
    '''



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



    # input_ids = torch.cat([torch.unsqueeze(ins[0], dim=0) for ins in data.data_test])
    # attention_mask = torch.cat([torch.unsqueeze(ins[1], dim=0) for ins in data.data_test])
    # labels = torch.cat([torch.tensor(np.expand_dims(ins[2], 0)) for ins in data.data_test])
    # print('input_ids', input_ids.size())  # torch.Size([10, 256])
    # print('attention_mask', attention_mask.size())  # torch.Size([10, 256])
    # print('labels', labels.size())  # torch.Size([10])

    TPLG = []
    PRED = []

    ''' 预测 '''
    model.eval()
    with torch.no_grad():

        # 把 dataset 放入 DataLoader
        # loader = Data.DataLoader(
        #     dataset=data.data_test,  # 数据，封装进Data.TensorDataset()类的数据
        #     batch_size=8,  # 每块的大小
        #     shuffle=False,  # 要不要打乱数据 (打乱比较好)
        #     num_workers=1,  # 多进程（multiprocess）来读数据
        # )
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
