import torch
import numpy as np

import torch.distributed as dist
from models.utils.comm import *
from .inferencer import eval_zs_gzsl
from apex import amp
import wandb
import datetime
import json

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,
        model_file_path,
        cfg,
    ):

    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    att_seen = res['att_seen'].to(device)
    att_unseen = res['att_unseen'].to(device)
    att = torch.cat((att_seen, att_unseen), dim=0)
    losses = []
    cls_losses = []
    reg_losses = []
    kl_losses = []

    bias_losses = []
    scale_all = []

    #======================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    wandb.init(project=cfg.PROJECT_NAME, config = cfg, name = nowtime, save_code=True)
    model.run_id = wandb.run.id
    #======================================================================

    model.train()

    for epoch in range(0, max_epoch):

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        kl_loss_epoch = []

        bias_loss_epoch = []
        scale_epoch = []

        scheduler.step()

        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):

            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            loss_dict = model(x=batch_img, att=batch_att, label=batch_label, seen_att=att_seen,att_all=att)


            Lreg = loss_dict['Reg_loss']
            Lcls = loss_dict['Cls_loss']
            #  add for parts loss
            Lbias = loss_dict['bias_loss']
            Lkl = loss_dict['kl_loss']
            scale = loss_dict['scale']
            loss_dict.pop('scale')

            loss = lamd[1]*Lcls+ lamd[2]*Lreg + lamd[3]*Lbias+ lamd[4]*Lkl
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            lreg = loss_dict_reduced['Reg_loss']
            lcls = loss_dict_reduced['Cls_loss']
            lbias = loss_dict_reduced['bias_loss']
            lkl = loss_dict_reduced['kl_loss']
            losses_reduced = lamd[1]*lcls + lamd[2]*lreg + lamd[3]*lbias+ lamd[4]*lkl
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

            loss_epoch.append(losses_reduced.item())
            cls_loss_epoch.append(lcls.item())
            reg_loss_epoch.append(lreg.item())
            bias_loss_epoch.append(lbias.item())
            kl_loss_epoch.append(lkl.item())
            scale_epoch.append(scale)


        if is_main_process():
            losses += loss_epoch
            cls_losses += cls_loss_epoch
            reg_losses += reg_loss_epoch
            bias_losses += bias_loss_epoch
            kl_losses += kl_loss_epoch
            scale_all += scale_epoch
            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            cls_loss_epoch_mean = sum(cls_loss_epoch) / len(cls_loss_epoch)
            reg_loss_epoch_mean = sum(reg_loss_epoch)/len(reg_loss_epoch)
            bias_loss_epoch_mean = sum(bias_loss_epoch) / len(bias_loss_epoch)
            kl_loss_epoch_mean = sum(kl_loss_epoch) / len(kl_loss_epoch)
            scale_epoch_mean = sum(scale_epoch) / len(scale_epoch)
            losses_mean = sum(losses) / len(losses)
            cls_losses_mean = sum(cls_losses) / len(cls_losses)
            reg_losses_mean = sum(reg_losses) / len(reg_losses)
            bias_losses_mean = sum(bias_losses) / len(bias_losses)
            kl_losses_mean = sum(kl_losses) / len(kl_losses)
            scale_all_mean = sum(scale_all) / len(scale_all)

            
            log_info = 'epoch: %d  |  loss: %.4f (%.4f), cls_loss: %.4f (%.4f), reg_loss: %.4f (%.4f),  bias_loss: %.4f (%.4f),  kl_loss: %.4f (%.4f),  scale:  %.4f (%.4f),    lr: %.6f' % \
                       (epoch + 1, loss_epoch_mean, losses_mean, cls_loss_epoch_mean, cls_losses_mean,reg_loss_epoch_mean,
                        reg_losses_mean, bias_loss_epoch_mean, bias_losses_mean,kl_loss_epoch_mean, kl_losses_mean,
                        scale_epoch_mean, scale_all_mean, optimizer.param_groups[0]["lr"])
            print(log_info)


        synchronize()
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))

            if H > best_performance[-1]:
                best_epoch=epoch+1
                best_performance[1:] = [acc_seen, acc_novel, H]
                data = {}
                data["model"] = model.state_dict()
                torch.save(data, model_file_path)
                print('save model: ' + model_file_path)

            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs
            
            wandb.log({'epoch':epoch, 'loss': losses_mean, 'cls_loss':cls_losses_mean,'reg_loss':reg_losses_mean,'bias_loss':bias_losses_mean,'kl_loss':kl_losses_mean,'lr':optimizer.param_groups[0]["lr"], \
                'zsl':acc_zs, 'gzsl_s':acc_seen,'gzsl_u':acc_novel,'H':H, \
                    'Best_zsl':best_performance[0],'Best_s':best_performance[1],'Best_u':best_performance[2],'Best_H':best_performance[3]})
            


    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % tuple(best_performance))
        wandb.finish()