from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



def vistensor_img(finalpred,name):
    demo=finalpred
    array = demo.detach().cpu().numpy()
    color_map=array.transpose(1, 2, 0)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    original_image = color_map * std + mean
    original_image = np.clip(original_image, 0, 255)  
    original_image = original_image.astype(np.uint8) 
    cv2.imwrite(name, original_image)



def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        try:
            # use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
            use_gpu = '0,1'
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)


class SemRunner(BaseRunner):

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)
        writer = None

        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)
    
        # train
        for iteration in range(cfg.max_iter):

            supportImgs,supportMasks,queryImgs,queryLabels,classNum= train_iterator.get()
            CatImgsTensor = torch.cat(supportImgs+queryImgs, dim=0).cuda()
            CatImgsTensor = CatImgsTensor.unsqueeze(0)
            supportMasks=torch.cat(supportMasks, dim=0).cuda()
            supportMasks = supportMasks.unsqueeze(0)
            queryLabels=torch.cat(queryLabels).cuda().long() 
            
            masks_pred, iou_pred= self.model(CatImgsTensor,supportMasks,queryLabels)
           
            total_loss = torch.zeros(1).cuda()
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, masks_pred, queryLabels,iou_pred, cfg)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(iteration=iteration, log_path=log_path, log_data=train_meter.get(clear=True),
                          status=self.exist_status[0],
                          writer=writer, timer=self.train_timer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:

                (FBIoU, _),(_, mIoU) = self._eval()

                model_path1=model_path[:-4]+'-'+str(iteration)+'.pth'
                save_model(self.model, model_path1, parallel=self.the_number_of_gpu > 1)
                print_and_save_log("saved model in {model_path}".format(model_path=model_path1), path=log_path)

                log_data = {'FBmIoU': FBIoU,'mIoU':mIoU}
                write_log(iteration=iteration, log_path=log_path, log_data=log_data, status=self.exist_status[1],
                          writer=writer, timer=self.eval_timer)

        save_model(self.model, model_path1, is_final=True, parallel=self.the_number_of_gpu > 1)
        if writer is not None:
            writer.close()


    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric_fb = mIoUOnline(class_names=class_names)
        eval_metric=mIoUOnline(class_names=[1,2,3,4,5])

        with torch.no_grad():
            for index, (supportImgs,supportMasks,queryImgs,queryLabels,classNum) in enumerate(self.val_loader):
                CatImgsTensor = torch.cat(supportImgs+queryImgs, dim=0).cuda()
                CatImgsTensor = CatImgsTensor.unsqueeze(0)
                supportMasks=torch.cat(supportMasks, dim=0).cuda()
                supportMasks = supportMasks.unsqueeze(0)
                queryLabels=torch.cat(queryLabels).cuda().long() 

                masks_pred, iou_pred = self.model(CatImgsTensor,supportMasks,queryLabels)

                predictions0 = torch.argmax(masks_pred[:,0:2,:,:], dim=1)
                predictions1 = torch.argmax(masks_pred[:,2:4,:,:], dim=1)
                predictions2 = torch.argmax(masks_pred[:,4:6,:,:], dim=1)
                predictions3 = torch.argmax(masks_pred[:,6:8,:,:], dim=1)
                pre=[predictions0,predictions1,predictions2,predictions3]
                best_idx = np.argmax(iou_pred.cpu().numpy())
                predictions=pre[best_idx]

                for batch_index in range(queryImgs[0].size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(queryLabels[batch_index].squeeze(0))
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric_fb.add(pred_mask, gt_mask)

                    pred_mask[np.where(pred_mask>0)]=classNum.item()
                    gt_mask[np.where(gt_mask>0)]=classNum.item()
                    eval_metric.add(pred_mask, gt_mask)


        self.model.train()
        return eval_metric_fb.get(clear=True),eval_metric.get(clear=True)

    def sum_of_differences(self,lst):
        if len(lst) < 2:
            return 0
        differences_sum = 0
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                difference = F.l1_loss(lst[j].float(), lst[i].float())
                differences_sum += difference
        return differences_sum

    def iou(self,preds, targets):
       
        intersection = (preds * targets).sum()   
        union = preds.sum() + targets.sum() - intersection  

        return intersection / union
    
    def spearman_ranking_loss(self,tensor1, tensor2):
        rank1 = tensor1.argsort().argsort().float() 
        rank2 = tensor2.argsort().argsort().float() 
        n = len(tensor1)
        diff = rank1 - rank2
        loss = torch.sum(diff * diff) / (n * (n*n - 1))
        return loss


    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels,iou_pred, cfg):

        tmp_loss1=F.cross_entropy(mask_pred[:,0:2,:,:], labels)
        tmp_loss2=F.cross_entropy(mask_pred[:,2:4,:,:], labels)
        tmp_loss3=F.cross_entropy(mask_pred[:,4:6,:,:], labels)
        tmp_loss4=F.cross_entropy(mask_pred[:,6:8,:,:], labels)
        total_loss += (tmp_loss1 +tmp_loss2+tmp_loss3+tmp_loss4)


        predictions0 = torch.argmax(mask_pred[:,0:2,:,:], dim=1)
        predictions1 = torch.argmax(mask_pred[:,2:4,:,:], dim=1)
        predictions2 = torch.argmax(mask_pred[:,4:6,:,:], dim=1)
        predictions3 = torch.argmax(mask_pred[:,6:8,:,:], dim=1)
        pre=[predictions0,predictions1,predictions2,predictions3]
        multiLoss=self.sum_of_differences(pre)
        total_loss +=multiLoss

   
        pre=[self.iou(predictions0.detach(),labels),self.iou(predictions1.detach(),labels),
             self.iou(predictions2.detach(),labels),self.iou(predictions3.detach(),labels)]
        pre=torch.tensor(pre).cuda()

        spearman_ranking_loss=self.spearman_ranking_loss(pre,iou_pred)
        total_loss +=spearman_ranking_loss*0.1

            

    def test(self,task_name,className=''):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric_fb = mIoUOnline(class_names=class_names)
        eval_metric=mIoUOnline(class_names=[1,2,3,4,5])

        with torch.no_grad():
            for index, (supportImgs,supportMasks,queryImgs,queryLabels,classNum) in enumerate(self.val_loader):
                CatImgsTensor = torch.cat(supportImgs+queryImgs, dim=0).cuda()
                CatImgsTensor = CatImgsTensor.unsqueeze(0)
                supportMasks=torch.cat(supportMasks, dim=0).cuda()
                supportMasks = supportMasks.unsqueeze(0)
                queryLabels=torch.cat(queryLabels).cuda().long()  

                masks_pred, iou_pred = self.model(CatImgsTensor,supportMasks,queryLabels)

                predictions0 = torch.argmax(masks_pred[:,0:2,:,:], dim=1)
                predictions1 = torch.argmax(masks_pred[:,2:4,:,:], dim=1)
                predictions2 = torch.argmax(masks_pred[:,4:6,:,:], dim=1)
                predictions3 = torch.argmax(masks_pred[:,6:8,:,:], dim=1)
                pre=[predictions0,predictions1,predictions2,predictions3]
                best_idx = np.argmax(iou_pred.cpu().numpy())
                predictions=pre[best_idx]

                for batch_index in range(queryImgs[0].size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(queryLabels[batch_index].squeeze(0))

                    pred_mask_ori=pred_mask.copy()
                    pred_mask_ori[np.where(pred_mask>0)]=255
                    pred_mask_ori = cv2.medianBlur(pred_mask_ori.astype(np.uint8), 3)

                    gt_mask_ori=gt_mask.copy()
                    gt_mask_ori[np.where(gt_mask>0)]=255

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric_fb.add(pred_mask, gt_mask)
                    pred_mask[np.where(pred_mask>0)]=classNum.item()
                    gt_mask[np.where(gt_mask>0)]=classNum.item()
                    eval_metric.add(pred_mask, gt_mask)

        return eval_metric_fb.get(clear=True),eval_metric.get(clear=True)