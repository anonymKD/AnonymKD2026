import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader, build_dataloader_ts
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger, set_determinism
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize


torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def main():
    args, args_text = parse_args()
    # print("args_text", args_text)
    args.exp_dir = f'{args.experiment_root}/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    # save args
    logger.info(args)
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    set_determinism(seed)


    '''build dataloader'''
    if args.input_modality == 'ts': #for time series data
        if args.val_size is not None:
            train_dataset, val_dataset,_, train_loader, val_loader, _= build_dataloader_ts(args)
        else:
            train_dataset, val_dataset, train_loader, val_loader = build_dataloader_ts(args)
    else: 
        train_dataset, val_dataset, train_loader, val_loader = build_dataloader(args)

    print("len(train_dataset):", len(train_dataset))
    print("len(train_loader):", len(train_loader))
    print("batch_size:", args.batch_size)
    print("world_size:", args.world_size if hasattr(args, "world_size") else "NA")

    '''build model'''
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()
    val_loss_fn = loss_fn
    # print("loss func-----------------------------",loss_fn )

    if args.train_transfer_head:
        # load pretrained student backbone
        args.num_classes=100
        args.pretrained_ckpt=args.pretrained_ckpt+ args.experiment +'/best.pth.tar'
        model = build_model(
            args,
            args.model,
            pretrained=args.student_pretrained,
            pretrained_ckpt=args.pretrained_ckpt
        )
        logger.info(model)

        in_features = model.fc.in_features

        if args.dataset == 'tinyimagenet':
            args.num_classes = 200
        elif args.dataset == 'stl10':
            args.num_classes = 10
        else:
            raise ValueError(f"Unknown transfer dataset: {args.dataset}")

        model.fc = nn.Linear(in_features, args.num_classes).cuda()

        if args.full_finetune:
            # full fine-tuning
            for param in model.parameters():
                param.requires_grad = True
        else:
            # freeze backbone, train only head
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc")
    else:
        model = build_model(args, args.model, args.student_pretrained, args.pretrained_ckpt)

    
    logger.info(f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M')

    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    model.cuda()
    
    #Train the diffsion and VAE
    if args.train_feature_space_diffsion:
        print("Training a  feature_space_diffsion----------")
        logger.info(model)
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt, args.teacher_model_config)
        logger.info(teacher_model)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)', is_latnet_diff=False)
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        from lib.models.losses.kd_loss import LatentDiffTrainLoss
        loss_fn = LatentDiffTrainLoss(model, teacher_model, args.teacher_model, args.generative_prior_kwargs, args.log_interval)
        val_loss_fn = loss_fn


    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt, args.teacher_model_config)
        logger.info(teacher_model)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        if args.generative_prior_kwargs != '' and args.kd in ['GVD']:
            gen_prior_ckpt = args.generative_prior_kwargs.get('generative_prior_ckpt')
            gen_prior_name = args.generative_prior_kwargs.get('generative_prior_model')
            gen_prior=build_model(args, gen_prior_name, pretrained=True, pretrained_ckpt=gen_prior_ckpt)
            args.generative_prior_kwargs['gen_prior'] = gen_prior
            
        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(model, teacher_model, args.model, args.teacher_model, loss_fn, 
                         args.kd, args.ori_loss_weight, args.kd_loss_weight, args.kd_loss_kwargs, 
                        generative_prior_kwargs=args.generative_prior_kwargs, total_epochs=args.epochs)

    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)
    loss_fn.student = model
    logger.info(model)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''build optimizer'''
    if args.train_transfer_head:
        trainable_params = [p for p in model.module.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=not args.sgd_no_nesterov,
        )
        print("\nTrainable parameters:")
        for name, param in model.module.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, shape={tuple(param.shape)}")
    else:
        optimizer = build_optimizer(args.opt,
                                    model.module,
                                    args.lr,
                                    eps=args.opt_eps,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    filter_bias_and_bn=not args.opt_no_filter,
                                    nesterov=not args.sgd_no_nesterov,
                                    sort_params=args.dyrep)
        

    '''build scheduler'''
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    print("steps_per_epoch-----",steps_per_epoch,"warmup_steps",warmup_steps,"decay_steps",decay_steps, "total_steps",total_steps)
    scheduler = build_scheduler(args.sched,
                                optimizer,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn
        dyrep = DyRep(
            model.module,
            optimizer,
            recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
                                           args.dyrep_recal_bn_iters, m),
            filter_bias_and_bn=not args.opt_no_filter)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'dyrep': dyrep
                                     }, mode=args.val_loss_monitor_mode)

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
        scheduler.step(start_epoch * len(train_loader))
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )
    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None

    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs

        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader,
                              optimizer, loss_fn, scheduler, auxiliary_buffer,
                              dyrep, loss_scaler)
        

        # validate
        test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
        if model_ema is not None:
            test_metrics = validate(args,
                                    epoch,
                                    model_ema.module,
                                    val_loader,
                                    val_loss_fn,
                                    log_suffix='(EMA)')

        # dyrep
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    # adjust
                    logger.info('DyRep: adjust model.')
                    dyrep.adjust_model()
                    logger.info(
                        f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
                    )
                    # re-init DDP
                    model = DDP(model.module,
                                device_ids=[args.local_rank],
                                find_unused_parameters=True)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info('DyRep: recalibrate BN.')
                    recal_bn(model.module, train_loader, 200)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics, score_key=args.val_loss_monitor_metric)
        logger.info('\n'.join(['Checkpoints:'] + [
            '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
        ]))

    
def train_epoch(args,
                epoch,
                model,
                model_ema,
                loader,
                optimizer,
                loss_fn,
                scheduler,
                auxiliary_buffer=None,
                dyrep=None,
                loss_scaler=None):
    loss_m = AverageMeter(dist=True)
    kd_loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    memory_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()

    # default: no KD loss (for teacher, base and LD training)
    kd_loss = torch.zeros(1)

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            input, target = batch
            idx = None
        else:
            input, target, idx = batch

        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=False) for speedup
        for p in model.parameters():
            p.grad = None

        with torch.cuda.amp.autocast(enabled=loss_scaler is not None):
            if args.kd:
                loss, kd_loss = loss_fn(input, target, epoch, idx)
            elif args.train_feature_space_diffsion:
                loss, _, diff_loss_only = loss_fn(input, target)
                kd_loss=diff_loss_only# for plotting only the diff loss
            else:
                output = model(input)
                loss = loss_fn(output, target)
    
            if auxiliary_buffer is not None:
                loss_aux = loss_fn(auxiliary_buffer.output, target)
                loss += loss_aux * auxiliary_buffer.loss_weight

        if loss_scaler is None:
            loss.backward()
        else:
            # amp
            loss_scaler.scale(loss).backward()
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)

        if dyrep is not None:
            # record states of model in dyrep
            dyrep.record_metrics()
            
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        kd_loss_m.update(kd_loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
        memory_m.update(peak_mem)


        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s '
                        .format(
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            lr=optimizer.param_groups[0]['lr'],
                            batch_time=batch_time_m,
                            data_time=data_time_m))
        scheduler.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()

    return {'train_loss': loss_m.avg, 'kd_loss': kd_loss_m.avg }


def validate(args, epoch, model, loader, loss_fn, log_suffix='', is_latnet_diff=True):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()
    
    model.eval()
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            input, target = batch
            idx = None
        else:
            input, target, idx = batch

        with torch.no_grad():
            if args.train_feature_space_diffsion and is_latnet_diff:
                loss, output, _diff_loss_only = loss_fn(input, target)
            else:
                output = model(input)
                loss = loss_fn(output, target)
            
        if args.num_classes >= 5:
            top1, top5 = accuracy(output, target, topk=(1, 5))
        else:
            top1, = accuracy(output, target, topk=(1,))
            top5 = top1
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.4f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}

if __name__ == '__main__':
    main()

