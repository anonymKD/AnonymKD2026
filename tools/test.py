import os
import torch
import torch.nn as nn
import logging
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.dataset.builder import build_dataloader, build_dataloader_ts
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger, set_determinism
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import csv
import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'{args.experiment_root}/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)


    '''fix random seed'''
    seed = args.seed + args.rank
    set_determinism(seed)

    '''build dataloader'''
    if args.input_modality == 'ts': #for time series data
        _train_dataset, _val_dataset, train_loader, val_loader = build_dataloader_ts(args)
    else: 
        _train_dataset, _val_dataset, train_loader, val_loader= build_dataloader(args)


    '''build model'''
    loss_fn = nn.CrossEntropyLoss().cuda()
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(model)
    logger.info(f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M')

    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    teacher_model=None

    #Test the diffsion and VAE
    if args.test_feature_space_diffsion:

        #test generated feature quality on train loader
        # val_loader=train_loader 

        logger.info(model)
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt, args.teacher_model_config)
        logger.info(teacher_model)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)', is_latent_diff=False)
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        from lib.models.losses.kd_loss import LatentDiffSampleLoss
        loss_fn = LatentDiffSampleLoss(model, teacher_model, args.teacher_model, args.generative_prior_kwargs, args.log_interval)
        val_loss_fn = loss_fn
    
    if args.test_student_model:
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt, args.teacher_model_config)
        logger.info(teacher_model)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank, mode=args.val_loss_monitor_mode)

    if args.resume:
        epoch = ckpt_manager.load(args.resume)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'epoch {epoch}'
        )
    else:
        epoch = 0

    # validate
    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn, teacher=teacher_model)
    if model_ema is not None:
        test_metrics = validate(args,
                                epoch,
                                model_ema.module,
                                val_loader,
                                loss_fn,
                                log_suffix='(EMA)')
    logger.info(test_metrics)

    log_metrics_to_csv(
    exp_dir=args.exp_dir,
    dataset=args.dataset,
    model=args.model,
    teacher=args.teacher_model,
    student_path=args.resume,
    metrics=test_metrics,
    prefix_filename=args.prefix_filename
    )


def validate(args, epoch, model, loader, loss_fn, log_suffix='', is_latent_diff=True, teacher=None):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()

    diff_samples_all = []
    latent_samples_all = []
    teacher_features_all=[]

    # for metrics
    all_targets = []
    all_probs = []
    all_preds = []
    all_logits = []
    all_preds_teacher = []
    all_logits_teacher=[]

    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            if args.test_feature_space_diffsion and is_latent_diff:
                #write here fileter all samples that teacher predictions are incorrect
                if args.collect_wrong == 1:
                    print("collect wrong labels---------------------###################----------------")
                    output_bk = teacher(input)
                    probs_teacher = F.softmax(output_bk, dim=1)
                    preds_teacher = torch.argmax(probs_teacher, dim=1)
                    wrong_mask = preds_teacher != target
                    input = input[wrong_mask]
                    target = target[wrong_mask]
                else:
                    input = input
                    target = target

                diff_samples, latent_samples, teacher_features, output_init, output, loss = loss_fn(input, target) # diff_samples_all, latent_samples_all, teacher_features_all, t_logits_init, t_logits_reconstructed, reconst_loss
                collect_all(diff_samples_all, diff_samples)
                collect_all(latent_samples_all, latent_samples)
                collect_all(teacher_features_all, teacher_features)

            else:
                output = model(input)
                loss = loss_fn(output, target)

            if teacher is not None:
                output_teacher = teacher(input)
                probs_teacher = F.softmax(output_teacher, dim=1)
                preds_teacher = torch.argmax(probs_teacher, dim=1)

            probs = F.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.append(target.detach().cpu())
            all_logits.append(output.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
            if teacher is not None:
                all_preds_teacher.append(preds_teacher.detach().cpu())
                all_logits_teacher.append(output_teacher.detach().cpu())

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
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

        # break;

    y_logits = torch.cat(all_logits, dim=0)
    y_true_t = torch.cat(all_targets, dim=0)
    y_true = y_true_t.numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()
  
    if teacher is not None:
        y_pred_teacher = torch.cat(all_preds_teacher, dim=0).numpy()
        y_logits_teacher = torch.cat(all_logits_teacher, dim=0)

        # Top-1 agreement (already yours)
        agreement_top1 = (y_pred == y_pred_teacher).mean() * 100

        # Top-5 agreement
        top5_student = torch.topk(y_logits, k=5, dim=1).indices  # [N, 5]
        teacher_top1 = torch.argmax(y_logits_teacher, dim=1, keepdim=True)  # [N, 1]

        agreement_top5 = (top5_student == teacher_top1).any(dim=1).float().mean().item() * 100

        # KL divergence
        kl_div = compute_kl(y_logits, y_logits_teacher)

        #write for file----
        teacher_wrong_mask = (y_pred_teacher != y_true)  # numpy bool [N]
        student_wrong_logits = y_logits[teacher_wrong_mask]          # torch [N_wrong, C]
        student_wrong_probs = torch.tensor(y_prob[teacher_wrong_mask]) # torch [N_wrong, C]
        teacher_wrong_logits = y_logits_teacher[teacher_wrong_mask]  # torch [N_wrong, C]
        targets_wrong = y_true_t[teacher_wrong_mask]                 # torch [N_wrong]

        save_path = os.path.join(args.exp_dir, args.prefix_filename + "_teacher_wrong_student_preds.pt")

        torch.save(
            {
                "student_logits": student_wrong_logits.cpu(),
                "student_probs": student_wrong_probs.cpu(),
                "teacher_logits": teacher_wrong_logits.cpu(),
                "targets": targets_wrong.cpu(),
                "teacher_preds": torch.tensor(y_pred_teacher[teacher_wrong_mask]).cpu(),
                "student_preds": torch.tensor(y_pred[teacher_wrong_mask]).cpu(),
                "teacher_wrong_mask": torch.tensor(teacher_wrong_mask).cpu(),
            },
            save_path
        )

        logger.info(f"Saved student prediction vectors on teacher-wrong samples to {save_path}")

    else:
        agreement_top1 = 0.0
        agreement_top5 = 0.0
        kl_div = 0.0
    
    if args.test_feature_space_diffsion and is_latent_diff:
        if args.save_prototypes == 1: # save prototype feature for correct smaples
            teacher_features = torch.cat(teacher_features_all, dim=0)
            teacher_logits = torch.cat(all_logits_teacher, dim=0)

            save_correct_class_prototypes(
                teacher_features=teacher_features,
                teacher_logits=teacher_logits,
                targets=y_true_t,
                save_path=args.exp_dir + "/teacher_correct_class_prototypes.pt",
                num_classes=args.num_classes,
            )
        #to measure how qualitative is the vicinal features in temrs of representing the correct class
        #clacluet mean target probabilites
        target_probs = y_prob[np.arange(len(y_true)), y_true]   # [N]
        mean_target_prob = target_probs.mean()
         #clacluet KL div with one hot labels
        kl = -np.log(target_probs + 1e-12)   # [N]
        kl_mean = kl.mean()

        # filter out anything that's not a tensor
        diff_samples_all = [x.detach().cpu() for x in diff_samples_all if torch.is_tensor(x)]
        latent_samples_all = [x.detach().cpu() for x in latent_samples_all if torch.is_tensor(x)]
        teacher_features_all = [x.detach().cpu() for x in teacher_features_all if torch.is_tensor(x)]

        diff_all = torch.cat(diff_samples_all, dim=0)      # [N, C, H, W]
        latent_all = torch.cat(latent_samples_all, dim=0)  # [N, C, H, W] 
        teacher_all = torch.cat(teacher_features_all, dim=0)  # [N, C, H, W] 

        #cosine similairty with orginal teacher features
        # flatten if needed
        if diff_all.dim() > 2:
            diff_flat = diff_all.flatten(1)
            teacher_flat = teacher_all.flatten(1)
        else:
            diff_flat = diff_all
            teacher_flat = teacher_all

        # normalize
        diff_norm = F.normalize(diff_flat, dim=1)
        teacher_norm = F.normalize(teacher_flat, dim=1)

        # cosine similarity
        cos_guided_teacher = (diff_norm * teacher_norm).sum(dim=1)  # [N]
        cos_with_orig_features=cos_guided_teacher.mean().item()
        print("Mean cos (guided, teacher):", cos_with_orig_features)

        #clalcuet the simlairty for protype classes-----
        # Load saved prototypes
        proto_data = torch.load(
            args.exp_dir + "/teacher_correct_class_prototypes.pt",
            map_location="cpu"
        )

        class_prototypes = proto_data["prototypes"]      # [num_classes, D]
        valid_classes = proto_data["valid_classes"]      # [num_classes]

        # Features to evaluate: use generated vicinal features
        z = diff_all   # or latent_all, depending on which one is your final guided feature

        # Flatten if feature maps: [N, C, H, W] -> [N, D]
        if z.dim() > 2:
            z = z.flatten(1)

        if class_prototypes.dim() > 2:
            class_prototypes = class_prototypes.flatten(1)

        # Normalize
        z = F.normalize(z, dim=1)
        class_prototypes = F.normalize(class_prototypes, dim=1)

        # Ground-truth labels
        targets = y_true_t.cpu().long()   # [N]

        # Only keep samples whose class prototype exists
        valid_mask = valid_classes[targets]

        z_valid = z[valid_mask]
        targets_valid = targets[valid_mask]

        # Cosine similarity to ground-truth prototype
        cos_gt = (z_valid * class_prototypes[targets_valid]).sum(dim=1)  # [N_valid]

        mean_cos_gt = cos_gt.mean().item()

        print("Mean cosine similarity to ground-truth prototype:", mean_cos_gt)

        save_per_sample_metrics(
            save_dir=args.exp_dir,
            prefix="with_guidance_2_100",
            targets=y_true_t,
            target_probs=target_probs,
            kl=kl,
            cos_gt=cos_gt,
            valid_mask=valid_mask,
        )
    
        ###############PLOT PLOT       
        print("teacher_features_all[0].shape-----", teacher_features_all[0].shape)
        print("input.shape-----", input.shape)


        logger.info(f'diff_all shape ({diff_all.shape}) latent_all shape: ({latent_all.shape}) labels_all shape: ({y_true.shape})')
        # plot_feature_space_2d(diff_all,   labels=y_true, method="tsne", title="Diffusion recon (t-SNE)", save_path=args.exp_dir+"/diff")
        # plot_feature_space_2d(latent_all, labels=y_true, method="tsne", title="Latent samples (t-SNE)",save_path=args.exp_dir+"/latent")
        # plot_feature_space_2d(teacher_all,labels=y_true, method="tsne",  title="Teacher features (t-SNE)", save_path=args.exp_dir+"/teacher")
        
        return {'reconstruction_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}
   
    # ---------------------------------------------------------------
    # Compute Expected Calbration Error and Brein Score, F1 and AUC-PRC
    # ----------------------------------------------------------------

    #Expected Calbration Error and Brein Score
    ece = compute_ece(y_logits, y_true_t)
    brier = compute_brier(y_logits, y_true_t)
    reliability_diagram(y_logits, y_true_t, n_bins=15, save_path=args.exp_dir+'/'+args.prefix_filename+'_relaibility.pdf', label='TeKAP')
    
    # F1
    f1_macro = f1_score(y_true, y_pred, average='macro')* 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted')* 100

    # AUC-PRC
    try:
        if args.num_classes == 2:
            # use probability of positive class
            auc_prc = average_precision_score(y_true, y_prob[:, 1])* 100
        else:
            # multiclass one-vs-rest macro AP
            classes = list(range(args.num_classes))
            y_true_bin = label_binarize(y_true, classes=classes)
            auc_prc = average_precision_score(y_true_bin, y_prob, average='macro')* 100
    except ValueError as e:
        logger.warning(f"Could not compute AUC-PRC: {e}")
        auc_prc = float('nan')

    logger.info(
        f'Test{log_suffix}: epoch={epoch} '
        f'loss={loss_m.avg:.4f} '
        f'top1={top1_m.avg:.3f} '
        f'top5={top5_m.avg:.3f} '
        f'f1_macro={f1_macro:.4f} '
        f'f1_weighted={f1_weighted:.4f} '
        f'auc_prc={auc_prc:.4f} '
        f'ece={ece:.4f} '
        f'brier={brier:.4f}'
        f'agreement_top1={agreement_top1:.4f}'
        f'agreement_top5={agreement_top5:.4f}'
        f'kl_div={kl_div:.4f}'
    )
    
    return {
        'test_loss': loss_m.avg,
        'top1': top1_m.avg,
        'top5': top5_m.avg,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_prc': auc_prc,
        'ece': ece,
        'brier': brier,
        'agreement_top1': agreement_top1,
        'agreement_top5': agreement_top5,
        "kl_div": kl_div
    }


#############################################################
# -----------------------------------------
# Support fucntions to calcualte test metrics and Visualize
# ----------------------------------------
def save_correct_class_prototypes(
    teacher_features,
    teacher_logits,
    targets,
    save_path,
    num_classes=100,
    normalize=True,
):
    """
    Compute class prototypes using only samples correctly predicted by the teacher.

    Args:
        teacher_features: Tensor [N, D] or [N, C, H, W]
        teacher_logits: Tensor [N, num_classes]
        targets: Tensor [N]
        save_path: path to save prototypes, e.g. "class_prototypes.pt"
        num_classes: number of classes. If None, inferred from teacher_logits.
        normalize: whether to L2-normalize prototypes before saving.

    Saves:
        dict with:
            prototypes: [num_classes, D]
            counts: [num_classes]
            valid_classes: [num_classes] bool
    """

    if teacher_features.dim() > 2:
        teacher_features = teacher_features.flatten(1)

    device = teacher_features.device
    targets = targets.to(device)
    teacher_logits = teacher_logits.to(device)

    if num_classes is None:
        num_classes = teacher_logits.size(1)

    preds = teacher_logits.argmax(dim=1)
    correct_mask = preds == targets

    features_correct = teacher_features[correct_mask]
    targets_correct = targets[correct_mask]

    D = features_correct.size(1)

    prototypes = torch.zeros(num_classes, D, device=device)
    counts = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        mask_c = targets_correct == c
        counts[c] = mask_c.sum()

        if counts[c] > 0:
            prototypes[c] = features_correct[mask_c].mean(dim=0)

    valid_classes = counts > 0

    if normalize:
        prototypes[valid_classes] = F.normalize(prototypes[valid_classes], dim=1)

    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "counts": counts.cpu(),
            "valid_classes": valid_classes.cpu(),
        },
        save_path,
    )

    print(f"Saved prototypes to {save_path}")
    print(f"Valid classes: {valid_classes.sum().item()} / {num_classes}")

def plot_pca_overlay_boundary_confidence_1(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=300000,
    seed=0,
    title="PCA feature-space overlay",
    score_mode="confidence",   # "confidence" or "entropy"
    point_size=18,
    alpha_min=0.35,
    alpha_max=0.95,
    figsize=(6, 5),
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=13,
    legend_fontsize=10,
    title_fontsize=14,
    boundary_alpha=0.18,
    boundary_linewidth=1.0,
    grid_step=0.05,
):
    """
    Combined PCA plot:
      - light background class regions from a 2D logistic regression in PCA space
      - points overlaid
      - point hue = class
      - point shade/alpha = confidence or entropy
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model
    device = next(net.parameters()).device

    # --------------------------------------------------
    # 1) Find final classifier layer
    # --------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    # --------------------------------------------------
    # 2) Hook penultimate features
    # --------------------------------------------------
    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # --------------------------------------------------
    # 3) Collect features, labels, scores
    # --------------------------------------------------
    feats_all = []
    labels_all = []
    scores_all = []

    collected = 0
    eps = 1e-12

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)

            logits = net(x)
            probs = F.softmax(logits, dim=1).cpu()

            if score_mode == "confidence":
                scores = probs.max(dim=1).values
            elif score_mode == "entropy":
                scores = -(probs * (probs + eps).log()).sum(dim=1)
            else:
                raise ValueError("score_mode must be 'confidence' or 'entropy'")

            feats = feat_buffer.pop()

            feats_all.append(feats)
            labels_all.append(y.detach().cpu())
            scores_all.append(scores)

            collected += feats.shape[0]
            if collected >= max_samples:
                break

    handle.remove()

    if len(feats_all) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(feats_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()
    scores = torch.cat(scores_all, dim=0).numpy()

    # truncate if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        scores = scores[idx]

    # --------------------------------------------------
    # 4) Select classes to plot
    # --------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    selected_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    scores = scores[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for a meaningful PCA plot.")

    # remap selected class ids to 0..K-1
    selected_classes_sorted = np.sort(selected_classes)
    class_map = {c: i for i, c in enumerate(selected_classes_sorted)}
    y_mapped = np.array([class_map[c] for c in y], dtype=np.int64)

    # --------------------------------------------------
    # 5) Normalize scores for shading
    # --------------------------------------------------
    if score_mode == "confidence":
        score_norm = np.clip(scores, 0.0, 1.0)
    else:
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            score_norm = np.ones_like(scores)
        else:
            score_norm = (scores - s_min) / (s_max - s_min)
        # invert so low entropy = darker, high entropy = lighter
        score_norm = 1.0 - score_norm

    # --------------------------------------------------
    # 6) PCA to 2D
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # --------------------------------------------------
    # 7) Fit 2D logistic regression for light background regions
    # --------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )
    clf.fit(X_2d, y_mapped)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # --------------------------------------------------
    # 8) Plot
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    # Stronger 8-color palette
    # strong_colors = [
    #     "#e41a1c",  # red
    #     "#377eb8",  # blue
    #     "#4daf4a",  # green
    #     "#984ea3",  # purple
    #     "#ff7f00",  # orange
    #     "#a65628",  # brown
    #     "#f781bf",  # pink
    #     "#17becf",  # cyan
    # ]
    strong_colors = [
            "#e41a1c",  # red
            "#377eb8",  # blue
            "#ff7f00",  # orange
            "#984ea3",  # purple
            "#4daf4a",  # green
            "#f781bf",  # pink
            "#a65628",  # brown
            "#17becf",  # cyan
            "#bcbd22",  # olive
            "#7f7f7f",  # gray
        ]

    if num_classes_to_plot > len(strong_colors):
        raise ValueError(
            f"This palette supports up to {len(strong_colors)} classes, "
            f"but got num_classes_to_plot={num_classes_to_plot}."
        )

    class_colors = strong_colors[:len(selected_classes_sorted)]
    cmap_bg = ListedColormap(class_colors)

    # light background regions
    plt.contourf(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        cmap=cmap_bg,
        alpha=boundary_alpha
    )

    # boundary lines
    plt.contour(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        colors="white",
        linewidths=boundary_linewidth,
        alpha=0.9
    )

    # overlay confidence/entropy-colored points
    for i, cls in enumerate(selected_classes_sorted):
        cls_mask = (y == cls)

        base_rgb = np.array(mcolors.to_rgb(class_colors[i]))
        cls_score = score_norm[cls_mask]

        colors = []
        alphas = alpha_min + (alpha_max - alpha_min) * cls_score

        for s in cls_score:
            shaded = (1.0 - s) * np.array([1.0, 1.0, 1.0]) + s * base_rgb
            colors.append(shaded)

        colors = np.array(colors)

        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            c=colors,
            s=point_size,
            alpha=alphas,
            edgecolors="none",
            label=f"class {cls}",
        )

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.grid(alpha=0.15)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA overlay plot to: {save_path}")

def plot_pca_overlay_boundary_confidence(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=300000,
    seed=0,
    title="PCA feature-space overlay",
    score_mode="confidence",   # "confidence" or "entropy"
    shade_mode="alpha",        # "alpha" or "whiten"
    point_size=18,
    alpha_min=0.35,
    alpha_max=0.95,
    figsize=(6, 5),
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=13,
    legend_fontsize=10,
    title_fontsize=14,
    boundary_alpha=0.18,
    boundary_linewidth=1.0,
    grid_step=0.05,
    use_fixed_palette=True,
):
    """
    Combined PCA plot:
      - light background class regions from a 2D logistic regression in PCA space
      - points overlaid
      - point hue = class
      - point shade/alpha = confidence or entropy

    score_mode:
      - "confidence": high confidence -> darker / stronger
      - "entropy": high entropy -> lighter / weaker

    shade_mode:
      - "alpha": keep class color fixed, vary transparency
      - "whiten": mix class color with white according to score
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model
    device = next(net.parameters()).device

    # --------------------------------------------------
    # 1) Find final classifier layer
    # --------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    # --------------------------------------------------
    # 2) Hook penultimate features
    # --------------------------------------------------
    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # --------------------------------------------------
    # 3) Collect features, labels, scores
    # --------------------------------------------------
    feats_all = []
    labels_all = []
    scores_all = []

    collected = 0
    eps = 1e-12

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)

            logits = net(x)
            probs = F.softmax(logits, dim=1).cpu()

            if score_mode == "confidence":
                scores = probs.max(dim=1).values
            elif score_mode == "entropy":
                scores = -(probs * (probs + eps).log()).sum(dim=1)
            else:
                raise ValueError("score_mode must be 'confidence' or 'entropy'")

            feats = feat_buffer.pop()

            feats_all.append(feats)
            labels_all.append(y.detach().cpu())
            scores_all.append(scores)

            collected += feats.shape[0]
            if collected >= max_samples:
                break

    handle.remove()

    if len(feats_all) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(feats_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()
    scores = torch.cat(scores_all, dim=0).numpy()

    # truncate if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        scores = scores[idx]

    # --------------------------------------------------
    # 4) Select classes to plot
    # --------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    selected_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    scores = scores[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for a meaningful PCA plot.")

    selected_classes_sorted = np.sort(selected_classes)
    class_map = {c: i for i, c in enumerate(selected_classes_sorted)}
    y_mapped = np.array([class_map[c] for c in y], dtype=np.int64)

    # --------------------------------------------------
    # 5) Normalize scores for shading
    # --------------------------------------------------
    if score_mode == "confidence":
        score_norm = np.clip(scores, 0.0, 1.0)
    else:
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            score_norm = np.ones_like(scores)
        else:
            score_norm = (scores - s_min) / (s_max - s_min)
        # invert so low entropy = stronger/darker, high entropy = weaker/lighter
        score_norm = 1.0 - score_norm

    # --------------------------------------------------
    # 6) PCA to 2D
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # --------------------------------------------------
    # 7) Fit 2D logistic regression for light background regions
    # --------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )
    clf.fit(X_2d, y_mapped)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # --------------------------------------------------
    # 8) Build a strong categorical palette
    # --------------------------------------------------
    if use_fixed_palette:
        # High-contrast categorical colors, good up to 8 classes
        fixed_palette = [
            "#e41a1c",  # red
            "#377eb8",  # blue
            "#ff7f00",  # orange
            "#984ea3",  # purple
            "#4daf4a",  # green
            "#f781bf",  # pink
            "#a65628",  # brown
            "#17becf",  # cyan
            "#bcbd22",  # olive
            "#7f7f7f",  # gray
        ]
        if num_classes_to_plot > len(fixed_palette):
            raise ValueError(
                f"Fixed palette supports up to {len(fixed_palette)} classes. "
                f"Got num_classes_to_plot={num_classes_to_plot}."
            )
        class_colors = fixed_palette[:len(selected_classes_sorted)]
    else:
        # fallback
        cmap = plt.get_cmap("tab20")
        class_colors = [mcolors.to_hex(cmap(i)) for i in range(len(selected_classes_sorted))]

    # background colormap for contourf
    bg_cmap = ListedColormap(class_colors)

    # --------------------------------------------------
    # 9) Plot
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    # light background regions
    plt.contourf(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        cmap=bg_cmap,
        alpha=boundary_alpha
    )

    # boundary lines
    plt.contour(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        colors="white",
        linewidths=boundary_linewidth,
        alpha=0.9
    )

    # overlay class-colored points
    for i, cls in enumerate(selected_classes_sorted):
        cls_mask = (y == cls)
        base_rgb = np.array(mcolors.to_rgb(class_colors[i]))
        cls_score = score_norm[cls_mask]

        alphas = alpha_min + (alpha_max - alpha_min) * cls_score

        if shade_mode == "alpha":
            # Keep strong distinct hue, vary alpha only
            plt.scatter(
                X_2d[cls_mask, 0],
                X_2d[cls_mask, 1],
                c=[base_rgb],
                s=point_size,
                alpha=alphas,
                edgecolors="none",
                label=f"class {cls}",
            )

        elif shade_mode == "whiten":
            # Mix with white according to score
            colors = []
            for s in cls_score:
                shaded = (1.0 - s) * np.array([1.0, 1.0, 1.0]) + s * base_rgb
                colors.append(shaded)
            colors = np.array(colors)

            plt.scatter(
                X_2d[cls_mask, 0],
                X_2d[cls_mask, 1],
                c=colors,
                s=point_size,
                alpha=alphas,
                edgecolors="none",
                label=f"class {cls}",
            )
        else:
            raise ValueError("shade_mode must be 'alpha' or 'whiten'")

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.grid(alpha=0.15)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA overlay plot to: {save_path}")

def plot_pca_overlay_boundary_confidence_bk(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=300000,
    seed=0,
    title="PCA feature-space overlay",
    score_mode="confidence",   # "confidence" or "entropy"
    point_size=18,
    alpha_min=0.35,
    alpha_max=0.95,
    figsize=(6, 5),
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=13,
    legend_fontsize=10,
    title_fontsize=14,
    boundary_alpha=0.18,
    boundary_linewidth=1.0,
    grid_step=0.05,
):
    """
    Combined PCA plot:
      - light background class regions from a 2D logistic regression in PCA space
      - points overlaid
      - point hue = class
      - point shade/alpha = confidence or entropy
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model
    device = next(net.parameters()).device

    # --------------------------------------------------
    # 1) Find final classifier layer
    # --------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    # --------------------------------------------------
    # 2) Hook penultimate features
    # --------------------------------------------------
    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # --------------------------------------------------
    # 3) Collect features, labels, scores
    # --------------------------------------------------
    feats_all = []
    labels_all = []
    scores_all = []

    collected = 0
    eps = 1e-12

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)

            logits = net(x)
            probs = F.softmax(logits, dim=1).cpu()

            if score_mode == "confidence":
                scores = probs.max(dim=1).values
            elif score_mode == "entropy":
                scores = -(probs * (probs + eps).log()).sum(dim=1)
            else:
                raise ValueError("score_mode must be 'confidence' or 'entropy'")

            feats = feat_buffer.pop()

            feats_all.append(feats)
            labels_all.append(y.detach().cpu())
            scores_all.append(scores)

            collected += feats.shape[0]
            if collected >= max_samples:
                break

    handle.remove()

    if len(feats_all) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(feats_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()
    scores = torch.cat(scores_all, dim=0).numpy()

    # truncate if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        scores = scores[idx]

    # --------------------------------------------------
    # 4) Select classes to plot
    # --------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    selected_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    scores = scores[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for a meaningful PCA plot.")

    # remap selected class ids to 0..K-1
    selected_classes_sorted = np.sort(selected_classes)
    class_map = {c: i for i, c in enumerate(selected_classes_sorted)}
    y_mapped = np.array([class_map[c] for c in y], dtype=np.int64)

    # --------------------------------------------------
    # 5) Normalize scores for shading
    # --------------------------------------------------
    if score_mode == "confidence":
        score_norm = np.clip(scores, 0.0, 1.0)
    else:
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            score_norm = np.ones_like(scores)
        else:
            score_norm = (scores - s_min) / (s_max - s_min)
        # invert so low entropy = darker, high entropy = lighter
        score_norm = 1.0 - score_norm

    # --------------------------------------------------
    # 6) PCA to 2D
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # --------------------------------------------------
    # 7) Fit 2D logistic regression for light background regions
    # --------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )
    clf.fit(X_2d, y_mapped)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # --------------------------------------------------
    # 8) Plot
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    base_cmap = plt.get_cmap("tab10")
    if num_classes_to_plot > 10:
        base_cmap = plt.get_cmap("tab20")

    # light background regions
    cmap_bg = plt.cm.get_cmap(base_cmap.name, len(selected_classes_sorted))
    plt.contourf(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        cmap=cmap_bg,
        alpha=boundary_alpha
    )

    # boundary lines
    plt.contour(
        xx, yy, Z,
        levels=np.arange(len(selected_classes_sorted) + 1) - 0.5,
        colors="white",
        linewidths=boundary_linewidth,
        alpha=0.9
    )

    # overlay confidence/entropy-colored points
    for i, cls in enumerate(selected_classes_sorted):
        cls_mask = (y == cls)

        base_rgb = np.array(base_cmap(i % base_cmap.N)[:3])
        cls_score = score_norm[cls_mask]

        colors = []
        alphas = alpha_min + (alpha_max - alpha_min) * cls_score

        for s in cls_score:
            shaded = (1.0 - s) * np.array([1.0, 1.0, 1.0]) + s * base_rgb
            colors.append(shaded)

        colors = np.array(colors)

        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            c=colors,
            s=point_size,
            alpha=alphas,
            edgecolors="none",
            label=f"class {cls}",
        )

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.grid(alpha=0.15)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA overlay plot to: {save_path}")
    
def plot_pca_class_confidence(
    model,
    loader,
    save_path,
    num_classes_to_plot=10,
    max_samples=30000,
    seed=0,
    point_size=18,
    alpha_min=0.35,
    alpha_max=0.95,
    figsize=(6, 5),
    title="Feature-space confidence",
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=13,
    legend_fontsize=10,
    title_fontsize=14,
    score_mode="confidence",   # "confidence" or "entropy"
):
    """
    Plot 2D PCA features where:
      - color (hue) indicates class
      - shade/alpha indicates confidence or entropy

    score_mode:
      - "confidence": high confidence -> darker/more saturated
      - "entropy": high entropy -> lighter/less saturated
    """
    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model
    device = next(net.parameters()).device

    # --------------------------------------------------
    # 1) Find final classifier layer
    # --------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    # --------------------------------------------------
    # 2) Hook penultimate features
    # --------------------------------------------------
    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # --------------------------------------------------
    # 3) Collect features, labels, scores
    # --------------------------------------------------
    feats_all = []
    labels_all = []
    scores_all = []

    collected = 0
    eps = 1e-12

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)

            logits = net(x)
            probs = F.softmax(logits, dim=1).cpu()

            if score_mode == "confidence":
                scores = probs.max(dim=1).values
            elif score_mode == "entropy":
                # entropy per sample
                scores = -(probs * (probs + eps).log()).sum(dim=1)
            else:
                raise ValueError("score_mode must be 'confidence' or 'entropy'")

            feats = feat_buffer.pop()

            feats_all.append(feats)
            labels_all.append(y.detach().cpu())
            scores_all.append(scores)

            collected += feats.shape[0]
            if collected >= max_samples:
                break

    handle.remove()

    if len(feats_all) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(feats_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()
    scores = torch.cat(scores_all, dim=0).numpy()

    # truncate if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        scores = scores[idx]

    # --------------------------------------------------
    # 4) Select classes to plot
    # --------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    selected_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    scores = scores[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for a meaningful PCA plot.")

    # --------------------------------------------------
    # 5) Normalize scores to [0, 1] for shading
    # --------------------------------------------------
    if score_mode == "confidence":
        # already roughly in [0, 1]
        score_norm = np.clip(scores, 0.0, 1.0)
    else:
        # entropy range depends on num_classes, normalize by observed range
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            score_norm = np.ones_like(scores)
        else:
            score_norm = (scores - s_min) / (s_max - s_min)

        # invert so low entropy = darker, high entropy = lighter
        score_norm = 1.0 - score_norm

    # --------------------------------------------------
    # 6) PCA to 2D
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # --------------------------------------------------
    # 7) Plot: hue = class, shade/alpha = score
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    base_cmap = plt.get_cmap("tab10")
    if num_classes_to_plot > 10:
        base_cmap = plt.get_cmap("tab20")

    selected_classes_sorted = np.sort(selected_classes)

    for i, cls in enumerate(selected_classes_sorted):
        cls_mask = (y == cls)

        base_rgb = np.array(base_cmap(i % base_cmap.N)[:3])
        cls_score = score_norm[cls_mask]

        colors = []
        alphas = alpha_min + (alpha_max - alpha_min) * cls_score

        for s in cls_score:
            # low normalized score -> lighter, high normalized score -> base color
            shaded = (1.0 - s) * np.array([1.0, 1.0, 1.0]) + s * base_rgb
            colors.append(shaded)

        colors = np.array(colors)

        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            c=colors,
            s=point_size,
            alpha=alphas,
            edgecolors="none",
            label=f"class {cls}",
        )

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.grid(alpha=0.2)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA class-{score_mode} plot to: {save_path}")

def plot_pca_class_confidence_bk(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=3000,
    seed=0,
    point_size=18,
    alpha_min=0.35,
    alpha_max=0.95,
    figsize=(6, 5),
    title="Feature-space confidence",
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=13,
    legend_fontsize=10,
    title_fontsize=14,
):
    """
    Plot 2D PCA features where:
      - color (hue) indicates class
      - shade/alpha indicates confidence

    This extracts the penultimate features by hooking the input to the final
    classifier layer (fc / classifier / head).

    Args:
        model: torch model (can be DDP-wrapped)
        loader: dataloader yielding (input, target)
        save_path: path to save figure (.pdf recommended)
        num_classes_to_plot: number of classes to visualize
        max_samples: maximum number of samples to collect
        seed: random seed
        point_size: scatter point size
        alpha_min: minimum alpha for low-confidence points
        alpha_max: maximum alpha for high-confidence points
        figsize: matplotlib figsize
    """
    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model
    device = next(net.parameters()).device

    # --------------------------------------------------
    # 1) Find final classifier layer
    # --------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    # --------------------------------------------------
    # 2) Hook penultimate features
    # --------------------------------------------------
    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # --------------------------------------------------
    # 3) Collect features, labels, confidences
    # --------------------------------------------------
    feats_all = []
    labels_all = []
    confs_all = []

    collected = 0

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)

            logits = net(x)
            probs = F.softmax(logits, dim=1).cpu()
            confs = probs.max(dim=1).values

            feats = feat_buffer.pop()

            feats_all.append(feats)
            labels_all.append(y.detach().cpu())
            confs_all.append(confs)

            collected += feats.shape[0]
            if collected >= max_samples:
                break

    handle.remove()

    if len(feats_all) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(feats_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()
    conf = torch.cat(confs_all, dim=0).numpy()

    # truncate if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        conf = conf[idx]

    # --------------------------------------------------
    # 4) Select classes to plot
    # --------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    selected_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    conf = conf[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for a meaningful PCA plot.")

    # --------------------------------------------------
    # 5) PCA to 2D
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # --------------------------------------------------
    # 6) Plot: hue = class, shade/alpha = confidence
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    # distinct class colors
    base_cmap = plt.get_cmap("tab10")
    if num_classes_to_plot > 10:
        base_cmap = plt.get_cmap("tab20")

    selected_classes_sorted = np.sort(selected_classes)

    for i, cls in enumerate(selected_classes_sorted):
        cls_mask = (y == cls)

        base_rgb = np.array(base_cmap(i % base_cmap.N)[:3])
        cls_conf = conf[cls_mask]

        # map low confidence -> lighter color, high confidence -> base color
        colors = []
        alphas = alpha_min + (alpha_max - alpha_min) * cls_conf

        for c in cls_conf:
            shaded = (1.0 - c) * np.array([1.0, 1.0, 1.0]) + c * base_rgb
            colors.append(shaded)

        colors = np.array(colors)

        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            c=colors,
            s=point_size,
            alpha=alphas,
            edgecolors="none",
            label=f"class {cls}",
        )

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.grid(alpha=0.2)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA class-confidence plot to: {save_path}")

def create_confidence_feature_plot(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=3000,
    seed=0,
    title="Feature-space confidence plot",
    color_mode="confidence",   # "confidence" or "entropy"
    point_size=18,
    alpha=0.85,
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    tick_fontsize=14,
    cbar_fontsize=14,
):
    """
    Plot 2D PCA features colored by model confidence or entropy.

    Args:
        model: torch model (can be DDP-wrapped)
        loader: dataloader
        save_path: output path (.pdf recommended)
        num_classes_to_plot: number of classes to visualize
        max_samples: maximum samples to collect
        seed: random seed
        title: plot title
        color_mode: "confidence" or "entropy"
    """
    rng = np.random.default_rng(seed)
    model.eval()

    net = model.module if hasattr(model, "module") else model

    # find classifier layer to hook penultimate features
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break
    if classifier is None:
        raise ValueError("Could not find classifier layer (fc/classifier/head).")

    feat_buffer = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    all_feats = []
    all_labels = []
    all_scores = []

    device = next(net.parameters()).device
    total_collected = 0

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            inp, target = batch[0], batch[1]
            inp = inp.to(device, non_blocking=True)

            logits = net(inp)
            probs = F.softmax(logits, dim=1).detach().cpu()

            feats = feat_buffer.pop()
            labels = target.detach().cpu()

            if color_mode == "confidence":
                scores = probs.max(dim=1).values
            elif color_mode == "entropy":
                eps = 1e-12
                scores = -(probs * (probs + eps).log()).sum(dim=1)
            else:
                raise ValueError("color_mode must be 'confidence' or 'entropy'")

            all_feats.append(feats)
            all_labels.append(labels)
            all_scores.append(scores)

            total_collected += feats.shape[0]
            if total_collected >= max_samples:
                break

    handle.remove()

    if len(all_feats) == 0:
        raise ValueError("No features collected.")

    X = torch.cat(all_feats, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    s = torch.cat(all_scores, dim=0).numpy()

    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y, s = X[idx], y[idx], s[idx]

    # choose classes with most samples
    unique_classes, counts = np.unique(y, return_counts=True)
    chosen_classes = unique_classes[np.argsort(-counts)[:num_classes_to_plot]]

    mask = np.isin(y, chosen_classes)
    X, y, s = X[mask], y[mask], s[mask]

    # PCA to 2D
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))

    # scatter points colored by confidence/entropy
    sc = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=s,
        cmap="viridis",
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    cbar = plt.colorbar(sc)
    if color_mode == "confidence":
        cbar.set_label("Confidence", fontsize=cbar_fontsize)
    else:
        cbar.set_label("Entropy", fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=max(cbar_fontsize - 2, 8))

    # overlay class markers lightly
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    for i, cls in enumerate(chosen_classes):
        cls_mask = (y == cls)
        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            facecolors='none',
            edgecolors='black',
            s=point_size + 8,
            linewidths=0.5,
            alpha=0.35,
            marker=markers[i % len(markers)],
            label=f"class {cls}"
        )

    plt.xlabel("PCA-1", fontsize=xlabel_fontsize)
    plt.ylabel("PCA-2", fontsize=ylabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.title(title, fontsize=xlabel_fontsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=10, frameon=False, loc="best", ncol=1)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved confidence feature plot to: {save_path}")

def create_decison_boundry_plot(
    model,
    loader,
    save_path,
    num_classes_to_plot=5,
    max_samples=3000,
    seed=0,
    title="Feature-space decision boundary"
):
    """
    Plot a 2D decision-boundary-style visualization in feature space.

    Steps:
      1. Extract penultimate features from the model
      2. Select a subset of classes
      3. Reduce features to 2D with PCA
      4. Fit a simple classifier in 2D
      5. Plot decision regions and samples

    Args:
        model: torch model (can be DDP-wrapped)
        loader: dataloader
        save_path: output figure path (.pdf or .png)
        num_classes_to_plot: number of classes to visualize
        max_samples: max number of samples to use
        seed: random seed
        title: figure title
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # ---------------------------------------------------------
    # 1) Get the actual module if DDP-wrapped
    # ---------------------------------------------------------
    net = model.module if hasattr(model, "module") else model

    # ---------------------------------------------------------
    # 2) Find the final classifier layer
    #    We capture its input = penultimate feature
    # ---------------------------------------------------------
    classifier = None
    for attr in ["fc", "classifier", "head"]:
        if hasattr(net, attr):
            classifier = getattr(net, attr)
            break

    if classifier is None or not isinstance(classifier, nn.Module):
        raise ValueError("Could not find final classifier layer (fc/classifier/head).")

    feat_buffer = []

    def pre_hook(module, inputs):
        # inputs is a tuple; input to final linear is penultimate feature
        x = inputs[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        feat_buffer.append(x.detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)

    # ---------------------------------------------------------
    # 3) Collect features and labels
    # ---------------------------------------------------------
    all_feats = []
    all_labels = []

    total_collected = 0
    device = next(net.parameters()).device

    with torch.no_grad():
        for batch in loader:
            # loader may return (input, target) or indexed variants
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    input, target = batch[0], batch[1]
                else:
                    continue
            else:
                continue

            input = input.to(device, non_blocking=True)
            _ = net(input)  # hook stores features

            feats = feat_buffer.pop()   # one entry per forward call
            labels = target.detach().cpu()

            all_feats.append(feats)
            all_labels.append(labels)

            total_collected += feats.shape[0]
            if total_collected >= max_samples:
                break

    handle.remove()

    if len(all_feats) == 0:
        raise ValueError("No features collected for decision boundary plot.")

    X = torch.cat(all_feats, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    # truncate if slightly over max_samples
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # ---------------------------------------------------------
    # 4) Select classes to visualize
    #    Use the most frequent classes among collected samples
    # ---------------------------------------------------------
    unique_classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    chosen_classes = unique_classes[order[:num_classes_to_plot]]

    mask = np.isin(y, chosen_classes)
    X = X[mask]
    y = y[mask]

    # remap class ids to 0..K-1 for plotting/classifier
    class_map = {c: i for i, c in enumerate(chosen_classes)}
    y_mapped = np.array([class_map[c] for c in y], dtype=np.int64)

    if len(np.unique(y_mapped)) < 2:
        raise ValueError("Need at least 2 classes to plot a decision boundary.")

    # ---------------------------------------------------------
    # 5) PCA to 2D
    # ---------------------------------------------------------
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    # ---------------------------------------------------------
    # 6) Fit a simple classifier in 2D
    # ---------------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )
    clf.fit(X_2d, y_mapped)

    # ---------------------------------------------------------
    # 7) Build mesh grid in 2D PCA space
    # ---------------------------------------------------------
    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0

    h = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # ---------------------------------------------------------
    # 8) Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 5))

    # background decision regions
    cmap_bg = plt.cm.get_cmap("tab10", len(chosen_classes))
    plt.contourf(xx, yy, Z, levels=np.arange(len(chosen_classes) + 1) - 0.5,
                 cmap=cmap_bg, alpha=0.25)

    # decision boundaries
    plt.contour(xx, yy, Z, levels=np.arange(len(chosen_classes) + 1) - 0.5,
                colors="white", linewidths=1.2, alpha=0.9)

    # sample points
    for original_class, mapped_class in class_map.items():
        cls_mask = (y_mapped == mapped_class)
        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            s=12,
            alpha=0.8,
            label=f"class {original_class}"
        )

    plt.xlabel("PCA-1", fontsize=14)
    plt.ylabel("PCA-2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc="best", frameon=False, ncol=1)
    # plt.title(title, fontsize=14)
    plt.grid(alpha=0.2)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved decision boundary plot to: {save_path}")

def collect_all(container, data):
    if isinstance(data, (list, tuple)):
        for x in data:
            collect_all(container, x)
    else:
        container.append(data)

#test metrics to measure student performances
def compute_kl(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean').item()

def compute_ece(logits, labels, n_bins=15):
    """
    logits: [N, num_classes]
    labels: [N]
    """
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        mask = (confidences > lower) & (confidences <= upper)

        if mask.sum() > 0:
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)

    return ece.item()

def compute_brier(logits, labels):
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(1)).float()
    return torch.mean((probs - one_hot) ** 2).item()

def reliability_diagram(
    logits,
    labels,
    n_bins=15,  #  reduce bins (important)
    save_path=None,
    label='Model',
    xlabel_fontsize=20,
    ylabel_fontsize=20,
    tick_fontsize=16,
    legend_fontsize=20,
    line_width=2.5,
    marker_size=6
):

    probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0, 1, n_bins + 1)

    accs = []
    confs = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.sum(mask) > 0:
            accs.append(np.mean(accuracies[mask]))
            confs.append(np.mean(confidences[mask]))
        else:
            accs.append(0)
            confs.append(0)

    #  control figure size (important for subfigures)
    plt.figure(figsize=(4, 4))

    #  thicker lines + bigger markers
    plt.plot(
        confs, accs,
        marker='o',
        markersize=marker_size,
        linewidth=line_width,
        label=label
    )

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        linewidth=line_width,
        label='Perfect'
    )

    plt.xlabel("Confidence", fontsize=xlabel_fontsize)
    plt.ylabel("Accuracy", fontsize=ylabel_fontsize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    #  compact legend
    plt.legend(fontsize=legend_fontsize, frameon=False)

    #  lighter grid
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved reliability diagram to: {save_path}")

    plt.close()


def plot_feature_space_2d(
    feats,
    labels=None,
    method="tsne",
    max_points=20000,
    pca_pre_reduce=50,
    perplexity=30,
    n_iter=1000,
    random_state=0,
    title=None,
    save_path=None,
    top_k_classes=10   # <-- NEW
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if not torch.is_tensor(feats):
        raise TypeError(f"feats must be a torch.Tensor, got {type(feats)}")

    x = feats.detach().cpu()
    if x.dim() > 2:
        x = x.flatten(1)
    X = x.numpy()

    y = None
    if labels is not None:
        y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"labels length {y.shape[0]} != N {X.shape[0]}")

        # -------- NEW: select top-k majority classes --------
        unique, counts = np.unique(y, return_counts=True)
        top_classes = unique[np.argsort(counts)[::-1][:top_k_classes]]

        mask = np.isin(y, top_classes)
        X = X[mask]
        y = y[mask]
        # ---------------------------------------------------

    # subsample
    n = X.shape[0]
    if max_points is not None and n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]

    # embed
    method_l = method.lower()
    if method_l == "pca":
        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method_l == "tsne":
        d = X.shape[1]
        k = min(pca_pre_reduce, d, X.shape[0] - 1)
        if k >= 2:
            Xr = PCA(n_components=k, random_state=random_state).fit_transform(X)
        else:
            Xr = X

        perpl = min(perplexity, max(5, (Xr.shape[0] - 1) // 3))

        emb = TSNE(
            n_components=2,
            perplexity=perpl,
            n_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(Xr)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # plot
    plt.figure(figsize=(6, 5))
    if y is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=8)
    else:
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=y, s=8)
        plt.colorbar(sc, fraction=0.046, pad=0.04)

    if title is None:
        title = f"{method.upper()} feature space (top-{top_k_classes} classes, n={emb.shape[0]})"
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return emb

def plot_feature_space_2d_bk(
    feats,                       # torch.Tensor [N,C,H,W] or [N,D]
    labels=None,                 # optional torch.Tensor/np.ndarray [N]
    method="tsne",               # "tsne" or "pca"
    max_points=20000,             # subsample for speed
    pca_pre_reduce=50,           # for t-SNE pre-reduction
    perplexity=30,
    n_iter=1000,
    random_state=0,
    title=None,
    save_path=None
):
    """
    Visualize feature tensor as 2D embedding using PCA or t-SNE.
    Designed for a single feature space at a time.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if not torch.is_tensor(feats):
        raise TypeError(f"feats must be a torch.Tensor, got {type(feats)}")

    x = feats.detach().cpu()
    if x.dim() > 2:
        x = x.flatten(1)                 # [N, C*H*W]
    X = x.numpy()

    y = None
    if labels is not None:
        y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"labels length {y.shape[0]} != N {X.shape[0]}")

    # subsample
    n = X.shape[0]
    if max_points is not None and n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]

    # embed
    method_l = method.lower()
    if method_l == "pca":
        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method_l == "tsne":
        # PCA pre-reduce for stability/speed
        d = X.shape[1]
        k = min(pca_pre_reduce, d, X.shape[0] - 1)
        if k >= 2:
            Xr = PCA(n_components=k, random_state=random_state).fit_transform(X)
        else:
            Xr = X

        # perplexity must be < n_samples
        perpl = min(perplexity, max(5, (Xr.shape[0] - 1) // 3))

        emb = TSNE(
            n_components=2,
            perplexity=perpl,
            n_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(Xr)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # plot
    plt.figure(figsize=(6, 5))
    if y is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=8)
    else:
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=y, s=8)
        plt.colorbar(sc, fraction=0.046, pad=0.04)

    if title is None:
        title = f"{method.upper()} feature space (n={emb.shape[0]})"
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return emb


def log_metrics_to_csv(exp_dir, dataset, model,teacher, student_path,metrics, prefix_filename='', filename="summary.csv"):
    os.makedirs(exp_dir, exist_ok=True)
    filename= prefix_filename + '_'+ filename
    csv_path = os.path.join(exp_dir, filename)

    # prepare row
    row = {
        "dataset": dataset,
        "teacher": teacher,
        "model": model,
        "student_path":student_path,
        **metrics
    }

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def compute_attention_map(feat, save_path=None, cmap='jet', use_softmax=False, tau=1.0):
    """
    feat: [C, H, W] or [1, C, H, W]
    returns: [H, W]
    """
    if feat.dim() == 4:
        feat = feat[0]   # [C, H, W]

    # Better spatial energy map
    x = feat.abs().mean(dim=0)   # [H, W]
    # alternatives:
    # x = feat.pow(2).mean(dim=0)
    # x = torch.norm(feat, p=2, dim=0)

    if use_softmax:
        H, W = x.shape
        x = F.softmax(x.view(-1) / tau, dim=0).view(H, W)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_np = x.detach().cpu().numpy()
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(x_np, cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return x



def overlay_channel_on_image(img, feat_map, alpha=0.5, cmap='jet', save_path=None):
    """
    img: [3, H, W] tensor in range [0,1] or normalized version you unnormalize first
    feat_map: [h, w] tensor for one channel
    """
    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    act = feat_map.detach().cpu()[None, None]  # [1,1,h,w]

    act_up = F.interpolate(
        act, size=img_np.shape[:2], mode='bilinear', align_corners=False
    )[0, 0].numpy()

    act_up = (act_up - act_up.min()) / (act_up.max() - act_up.min() + 1e-8)

    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(img_np, 0, 1))
    plt.imshow(act_up, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    


def overlay_all_channels_on_image(
    img,
    feat,
    alpha=0.5,
    cmap='jet',
    save_path=None,
    max_cols=8,
    unnormalize_fn=None,
    show_original_first=True,
):
    """
    img: [3, H, W]
    feat: [C, h, w] or [1, C, h, w]
    alpha: overlay strength
    max_cols: number of columns in grid
    unnormalize_fn: optional function(img)->img for de-normalization
    show_original_first: whether to place original image in first panel
    """

    if feat.dim() == 4:
        feat = feat[0]   # -> [C, h, w]

    if img.dim() != 3:
        raise ValueError(f"img must be [3,H,W], got {tuple(img.shape)}")
    if feat.dim() != 3:
        raise ValueError(f"feat must be [C,h,w] or [1,C,h,w], got {tuple(feat.shape)}")

    img = img.detach().cpu()
    feat = feat.detach().cpu()

    if unnormalize_fn is not None:
        img = unnormalize_fn(img)

    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()

    C, h, w = feat.shape
    n_panels = C + (1 if show_original_first else 0)

    cols = min(max_cols, n_panels)
    rows = math.ceil(n_panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    panel_idx = 0

    if show_original_first:
        axes[panel_idx].imshow(np.clip(img_np, 0, 1))
        axes[panel_idx].set_title("original", fontsize=8)
        axes[panel_idx].axis("off")
        panel_idx += 1

    for c in range(C):
        ax = axes[panel_idx]

        act = feat[c].unsqueeze(0).unsqueeze(0)   # [1,1,h,w]
        act_up = F.interpolate(
            act, size=img_np.shape[:2], mode='bilinear', align_corners=False
        )[0, 0].numpy()

        act_up = (act_up - act_up.min()) / (act_up.max() - act_up.min() + 1e-8)

        ax.imshow(np.clip(img_np, 0, 1))
        ax.imshow(act_up, cmap=cmap, alpha=alpha)
        ax.set_title(f"ch {c}", fontsize=8)
        ax.axis("off")
        panel_idx += 1

    for k in range(panel_idx, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)

    plt.close(fig)



##############################
#############################


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def unnormalize_cifar100(img):
    """
    img: [3, H, W]
    returns image in [0,1]
    """
    mean = torch.tensor(CIFAR100_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=img.device).view(3, 1, 1)
    img = img * std + mean
    return torch.clamp(img, 0, 1)


def feature_energy_map(feat, method="l2"):
    """
    feat: [C, H, W] or [1, C, H, W]
    returns: [H, W]
    """
    if feat.dim() == 4:
        feat = feat[0]

    if method == "l2":
        m = feat.pow(2).sum(dim=0).sqrt()
    elif method == "mean_abs":
        m = feat.abs().mean(dim=0)
    elif method == "mean_sq":
        m = feat.pow(2).mean(dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return m


def normalize_map(m):
    m = m - m.min()
    m = m / (m.max() - m.min() + 1e-8)
    return m


def upsample_map(m, out_hw=(32, 32)):
    """
    m: [H, W]
    """
    m = m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    return m[0, 0]


def save_cifar100_feature_triplet(
    img,
    feat_student,
    feat_denoised,
    feat_teacher,
    save_path,
    method="l2",
    cmap="viridis",
):
    """
    img: [3, 32, 32]
    feat_student: [C, h, w]
    feat_denoised: [C, h, w]
    feat_teacher: [C, h, w]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = unnormalize_cifar100(img.detach().cpu())
    img_np = img.permute(1, 2, 0).numpy()

    ms = feature_energy_map(feat_student.detach().cpu(), method=method)
    md = feature_energy_map(feat_denoised.detach().cpu(), method=method)
    mt = feature_energy_map(feat_teacher.detach().cpu(), method=method)

    ms = normalize_map(upsample_map(ms, (32, 32))).numpy()
    md = normalize_map(upsample_map(md, (32, 32))).numpy()
    mt = normalize_map(upsample_map(mt, (32, 32))).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    axes[0].imshow(np.clip(img_np, 0, 1))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(ms, cmap=cmap)
    axes[1].set_title("Student")
    axes[1].axis("off")

    axes[2].imshow(md, cmap=cmap)
    axes[2].set_title("Denoised student")
    axes[2].axis("off")

    axes[3].imshow(mt, cmap=cmap)
    axes[3].set_title("Teacher")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def save_cifar100_feature_grid(
    imgs,
    feats_student,
    feats_denoised,
    feats_teacher,
    save_path,
    n_rows=8,
    method="l2",
    cmap="viridis",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert lists of tensors to one tensor if needed
    if isinstance(imgs, list):
        imgs = torch.cat(imgs, dim=0)
    if isinstance(feats_student, list):
        feats_student = torch.cat(feats_student, dim=0)
    if isinstance(feats_denoised, list):
        feats_denoised = torch.cat(feats_denoised, dim=0)
    if isinstance(feats_teacher, list):
        feats_teacher = torch.cat(feats_teacher, dim=0)

    # Safety checks
    print("imgs.shape:", tuple(imgs.shape))
    print("feats_student.shape:", tuple(feats_student.shape))
    print("feats_denoised.shape:", tuple(feats_denoised.shape))
    print("feats_teacher.shape:", tuple(feats_teacher.shape))

    n_rows = min(
        n_rows,
        imgs.shape[0],
        feats_student.shape[0],
        feats_denoised.shape[0],
        feats_teacher.shape[0],
    )

    fig, axes = plt.subplots(n_rows, 4, figsize=(10, 2.4 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_rows):
        img = unnormalize_cifar100(imgs[i].detach().cpu())
        img_np = img.permute(1, 2, 0).numpy()

        ms = feature_energy_map(feats_student[i].detach().cpu(), method=method)
        md = feature_energy_map(feats_denoised[i].detach().cpu(), method=method)
        mt = feature_energy_map(feats_teacher[i].detach().cpu(), method=method)

        ms = normalize_map(upsample_map(ms, (32, 32))).numpy()
        md = normalize_map(upsample_map(md, (32, 32))).numpy()
        mt = normalize_map(upsample_map(mt, (32, 32))).numpy()

        axes[i, 0].imshow(np.clip(img_np, 0, 1))
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ms, cmap=cmap)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(md, cmap=cmap)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mt, cmap=cmap)
        axes[i, 3].axis("off")

    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Teacher original")
    axes[0, 2].set_title("Diff Generated")
    axes[0, 3].set_title("Teacher Latent")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_per_sample_metrics(
    save_dir,
    prefix,
    targets,
    target_probs,
    kl,
    cos_gt=None,
    valid_mask=None,
):
    """
    Save per-sample metrics so that init/guided deltas can be computed later.

    Saves:
        targets: [N]
        target_probs: [N]
        kl: [N]
        cos_gt: [N] if valid_mask is None, otherwise [N] with NaN for invalid classes
    """

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(target_probs, np.ndarray):
        target_probs = torch.from_numpy(target_probs)
    if isinstance(kl, np.ndarray):
        kl = torch.from_numpy(kl)

    targets = targets.detach().cpu().long()
    target_probs = target_probs.detach().cpu().float()
    kl = kl.detach().cpu().float()

    save_dict = {
        "targets": targets,
        "target_probs": target_probs,
        "kl": kl,
    }

    if cos_gt is not None:
        cos_gt = cos_gt.detach().cpu().float()

        if valid_mask is not None:
            valid_mask = valid_mask.detach().cpu().bool()

            cos_full = torch.full(
                (targets.size(0),),
                float("nan"),
                dtype=torch.float32,
            )
            cos_full[valid_mask] = cos_gt
            save_dict["cos_gt"] = cos_full
            save_dict["valid_mask"] = valid_mask
        else:
            save_dict["cos_gt"] = cos_gt

    save_path = os.path.join(save_dir, f"{prefix}_per_sample_metrics.pt")
    torch.save(save_dict, save_path)

    print(f"Saved per-sample metrics to: {save_path}")




if __name__ == '__main__':
    main()
