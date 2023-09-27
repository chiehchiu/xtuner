# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import os.path as osp
from functools import partial

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments

from xtuner.configs import cfgs_name_path
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.model.fast_forward import dispatch_fast_forward
from xtuner.model.utils import LoadWoInit, find_all_linear_names, traverse_dict
from xtuner.registry import BUILDER

def parse_args():
    parser = argparse.ArgumentParser(description='Train LLMs arguments')
    # In Python, the help key in the parser.add_argument('--work-dir', help='the dir to save logs and models') code snippet is used to provide a brief description of the argument. When the user runs the program with the --help option, argparse will print the help message, which includes the description of each argument. In this case, the help message will include the description "the dir to save logs and models" for the --work-dir argument. This is a useful feature for users who are not familiar with the program and need guidance on how to use it.
    parser.add_argument('config', help='config file name or path. Note: Please use the original '
        'configs, instead of the automatically saved log configs.')
    parser.add_argument('--work-dir', help='the dir to save logs and models') #code snippet
    parser.add_argument('--deepspeed', type=str, default=None, help='the path to the .json file for deepspeed')
    parser.add_argument('--resume', type=str, default=None, help='specify checkpoint files resumed from')
    #在这种情况下，nargs='+'表示将使用一个或多个参数。Config 用于读取配置文件, DictAction 将命令行字典类型参数转化为 key-value 形式
    # 其他参数: 可以使用 --cfg-options a=1,2,3 指定其他参数
    # 参数结果: {'a': [1, 2, 3]}
    parser.add_argument('--cfg-options', nargs='+',action=DictAction,help='override some settings in the used config')
    parser.add_argument('--launcher',choices=['none','pytorch','slurm','mpi'],default='none',help='job launcher')
    parser.add_argument('--local_rank','--local-rankl',type=int,default=0)
    # 没有使用递归。它只是调用解析器对象的parse_args()方法来解析用户提供的命令行参数，并将它们存储在args变量中。
    # 这边只传了一个变量config file的地址;
    args = parser.parse_args()
    return args


def main():    
    # parse config
    # deepspeed=None的使用     
    args = parse_args()

    if not os.path.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')
    
    #load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # checking if the value of the 'framework'
    # skip:mmengine
    # d.get(‘key’,‘value’):如果字典中存在关键字key，则返回关键字对应的值；如果字典d中不存在关键字key，则返回value的值
    if cfg.get('framework','mmengine').lower() == 'huggingface':
        # set deafult training_args
        if cfg.get('training_args', None) is None:
            cfg.training_args = dict(type=TrainingArguments)
        # set work_dir
        if args.work_dir is not None:
            cfg.training_args.output_dir =args.work_dir
         # elif cfg.get('output_dir', None ) is None:
        elif cfg.training_args.get('output_dir', None) is None:
            cfg.training_args.output_dir = osp.join(
                './work_dirs',
                osp.splitext(osp.basename(args.config))[0]
            )
        # build training_args
        training_args = BUILDER.build(cfg.training_args)
        # build models
        device_map ={'':int(os.environ.get('LOCAL_RANK', args.local_rank))}
        with LoadWoInit():
            cfg.model.device_map = device_map
            # 遍历model
            traverse_dict(cfg.model)
            model = BUILDER.build(cfg.model)
        model.config.use_cache = False
        if cfg.get('lora',None):
            lora = BUILDER.build(cfg.lora)
            model = prepare_model_for_kbit_training(model)
            if lora.target_module is None:
                modules = find_all_linear_names(model)
                lora.target_modules = modules
            model  = get_peft_model(model, lora)
        dispatch_fast_forward(model)
        #build dataset
        train_dataset = BUILDER.build(cfg.train_dataset)
        data_collator = partial(default_collate_fn, return_hf_format=True)
        #build trainer
        trainer = cfg.trainer(
            model = model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        # training
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    else:
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        cfg.launcher = args.launcher
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir',None) is None:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        
        if args.deepspeed:
            try:
                import deepspeed
            except ImportError:
                raise ImportError(
                    'deepspeed is not installed properly, please check.'
                )
            optim_wrapper = cfg.optim_wrapper.type
            # DeepSpeedOptimWrapper这边是直接通过参数进行调用，而不是以json的形式
            if optim_wrapper is not None == 'DeepSpeedOptimWrapper':
                print_log(
                    'Deepspeed training is already enabled in your configuration',
                    logger='current',
                    level=logging.WARNING)
            else:
                if not os.path.isfile(args.deepspeed):
                    try:
                        args.deepspeed = cfgs_name_path.split[args.deepspeed]
                    except KeyError:
                        raise FileNotFoundError(
                            f'Cannot find {args.deepspeed}')
                with open(args.deepspeed) as f:
                    ds_cfg = json.load(f)

                # gradient_accumulation_steps
                ds_grad_accum = ds_cfg.get('gradient_accumulation_steps', 'auto')
                
                mm_grad_accum = cfg.optim_wrapper.get('accumulative_counts',1)
                if ds_grad_accum != 'auto' and ds_grad_accum != mm_grad_accum:
                    print_log('Mismatch on grad_accumulation_steps:'
                                 f'MMEngine {mm_grad_accum},'
                                 f'Deepspeed {ds_grad_accum}. '
                                 f'Set to {mm_grad_accum}',
                                logger='current',
                                level=logging.WARNING)
                grad_accum = mm_grad_accum

                ds_train_bs = ds_cfg.get('train_micro_batch_size_per_gpu','auto')
                mm_train_bs = cfg.train_dataloader.batch_size 
                if ds_train_bs != 'auto' and ds_train_bs != mm_train_bs:
                    print_log(
                        ('Mismatch on train_micro_batch_size_per_gpu: '
                         f'MMEngine {mm_train_bs}, Deepspeed {ds_train_bs}. '
                         f'Set to {mm_train_bs}'),
                        logger='current',
                        level=logging.WARNING)
                train_bs = cfg.train_dataloader.batch_size
                
                ds_grad_clip = ds_cfg.get('gradient_clipping', 'auto')
                clip_grad = cfg.optim_wrapper.get('clip_grad', None)
                if clip_grad and clip_grad.get('max_norm'):
                    mm_max_norm = cfg.optim_wrapper.clip_grad.max_norm
                else:
                    mm_max_norm = 1.0
                if ds_grad_clip != 'auto' and ds_grad_clip != mm_max_norm:
                    print_log(
                        ('Mismatch on gradient_clipping: '
                         f'MMEngine {mm_max_norm}, Deepspeed {ds_grad_clip}. '
                         f'Set to {mm_max_norm}'),
                        logger='current',
                        level=logging.WARNING)
                grad_clip = mm_max_norm
                strategy = dict(
                    type='DeepSpeedStrategy',
                    config=args.deepspeed,
                    gradient_accumulation_steps=grad_accum,
                    train_micro_batch_size_per_gpu=train_bs,
                    gradient_clipping=grad_clip)
                cfg.__setitem__('strategy', strategy)
                optim_wrapper = dict(
                    type='DeepSpeedOptimWrapper',
                    optimizer=cfg.optim_wrapper.optimizer)
                cfg.__setitem__('optim_wrapper', optim_wrapper)
                cfg.runner_type = 'FlexibleRunner'
        # resume from 
        if args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        #build the runner from config
        if 'runner_type' not in cfg:
            #build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()

if __name__ == '__main__':
    main()
