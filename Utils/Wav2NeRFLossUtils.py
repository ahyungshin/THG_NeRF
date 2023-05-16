import os
import cv2
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.nn.functional as nnf
from collections import OrderedDict

from NetWorks.dwt import multi_level_dwt
from Utils.syncnet import SyncNet_color as SyncNet


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

# modified
class SimCLR(object):

    def __init__(self, device):
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.temperature = 0.5

    def info_nce_loss(self, features_a, features_v):

        features_a = F.normalize(features_a, dim=1)
        features_v = F.normalize(features_v, dim=1)

        similarity_matrix = torch.matmul(features_v, features_a.T) # [1,8]
        logits = similarity_matrix / self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        return logits, labels


class Wav2NeRFLossUtils(object):

    def __init__(self, model_path, bg_type = "white", use_vgg_loss = True, device = None) -> None:
        super().__init__()

        if bg_type == "white":
            self.bg_value = 1.0
        elif bg_type == "black":
            self.bg_value = 0.0
        else:
            self.bg_type = None
            print("Error BG type. ")
            exit(0)

        self.use_vgg_loss = use_vgg_loss
        if self.use_vgg_loss:
            assert device is not None
            self.device = device
            self.vgg_loss_func = VGGPerceptualLoss(resize = True).to(self.device)


        # Syncnet loss
        self.syncnet_T=5
        self.syncnet = SyncNet().to(self.device)
        self.syncnet.eval()
        state_dict = torch.load(os.path.join(model_path, "lipsync_expert.pth"), map_location=torch.device("cpu"))["state_dict"]
        self.syncnet.load_state_dict(state_dict)
        for p in self.syncnet.parameters():
            p.requires_grad = False

        self.recon_loss = nn.L1Loss()
        self.simclr_loss = SimCLR(device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


    @staticmethod
    def calc_cam_loss(delta_cam_info):
        delta_eulur_loss = torch.mean(delta_cam_info["delta_eulur"] * delta_cam_info["delta_eulur"])
        delta_tvec_loss = torch.mean(delta_cam_info["delta_tvec"] * delta_cam_info["delta_tvec"])

        return {
            "delta_eular": delta_eulur_loss,
            "delta_tvec": delta_tvec_loss
        }


    def calc_code_loss(self, opt_code_dict):

        iden_code_opt = opt_code_dict["iden"]
        expr_code_opt = opt_code_dict["expr"]

        iden_loss = torch.mean(iden_code_opt * iden_code_opt)
        expr_loss = torch.mean(expr_code_opt * expr_code_opt)

        appea_loss = torch.mean(opt_code_dict["appea"] * opt_code_dict["appea"])

        res_dict = {
            "iden_code":iden_loss,
            "expr_code":expr_loss,
            "appea_code":appea_loss,
        }

        return res_dict


    def get_sync_loss(self, mel, g): # [5,3,512,512]
        g = g[:, :, g.size(3)//2:, :].permute(0,1,3,2)  # [5,3,512,256]
        g = g.unsqueeze(0).permute(0,2,1,4,3)
        g = torch.cat([g[:, :, i] for i in range(self.syncnet_T)], dim=1)

        feats_a = [None] * mel.shape[0]
        for num in range(mel.shape[0]):
            a, v = self.syncnet(mel[num].unsqueeze(0).unsqueeze(0), g)
            feats_a[num] = a
        feats_a = torch.cat(feats_a, dim=0) # a: [8,512], v: [1,512]

        logits, labels = self.simclr_loss.info_nce_loss(feats_a, v)
        loss = self.criterion(logits, labels)
        return loss


    def calc_data_loss(self, iter_, data_dict, gt_rgb, gt_rect, gt_mel, head_mask_c1b, nonhead_mask_c1b):

        res_img = data_dict["merge_img"]
        head_mask_c3b = head_mask_c1b.expand(-1, 3, -1, -1)
        head_loss = F.mse_loss(res_img[head_mask_c3b], gt_rgb[head_mask_c3b])

        nonhead_mask_c3b = nonhead_mask_c1b.expand(-1, 3, -1, -1)
        nonhead_loss = F.mse_loss(res_img[nonhead_mask_c3b], gt_rgb[nonhead_mask_c3b])

        # Multi-Level SyncNet Loss
        sync_loss = 0.0
        if iter_ < 40000:
            sync_loss = 0.0
        else:
            fused_feature = data_dict["sync_mid"]
            cropped_list = [None] * fused_feature.shape[0]
            cropped_last_lst = [None] * res_img.shape[0]

            scale = fused_feature.shape[2]/gt_rgb.shape[2]

            for bs in range(res_img.shape[0]):
                x = int(gt_rect[bs,0])
                y = int(gt_rect[bs,1])
                w = int(gt_rect[bs,2])
                h = int(gt_rect[bs,3])

                cropped_last = res_img[bs][:, y:y+h, x:x+w].unsqueeze(0)
                cropped_last = nnf.interpolate(cropped_last, size=(96, 96), mode='bicubic', align_corners=False)
                cropped_last_lst[bs] = cropped_last

                resized_x = int(x * scale)
                resized_y = int(y * scale)
                resized_h = int(h * scale)
                resized_w = int(w * scale)

                cropped = fused_feature[bs][:, resized_y:resized_y+resized_h, resized_x:resized_x+resized_w].unsqueeze(0)
                cropped = nnf.interpolate(cropped, size=(96, 96), mode='bicubic', align_corners=False)
                cropped_list[bs] = cropped

            cropped_list = torch.cat(cropped_list, dim=0)
            cropped_last_lst = torch.cat(cropped_last_lst, dim=0)
            sync_loss = self.get_sync_loss(gt_mel, cropped_list) + self.get_sync_loss(gt_mel, cropped_last_lst)# mel:[1,5,16,80]
            sync_loss *= 0.03

        # DWT loss
        dwt_gt = multi_level_dwt(gt_rgb, levels=3) # [5,12,256]
        dwt = data_dict["dwt"]
        dwt_loss = 0.
        for dd in range(len(dwt)):
            # dwt_loss += self.recon_loss(dwt[dd], dwt_gt[dd])
            dwt_loss += 1 * self.recon_loss(dwt[dd][:,:3,:,:], dwt_gt[dd][:,:3,:,:]) # Lv.1
            dwt_loss += 10 * self.recon_loss(dwt[dd][:,3:,:,:], dwt_gt[dd][:,3:,:,:]) # Lv.2~4

        res = {
            "head_loss": head_loss,
            "nonhaed_loss": 0.01 * nonhead_loss,
            "dwt_loss": dwt_loss,
            "sync_loss": sync_loss
        }

        if self.use_vgg_loss:
            masked_gt_img = gt_rgb.clone()
            temp_res_img = res_img
            vgg_loss = self.vgg_loss_func(temp_res_img, masked_gt_img)
            res["vgg"] = vgg_loss

        return res


    def calc_total_loss(self, iter_, delta_cam_info, opt_code_dict, pred_dict, gt_rgb, gt_rect, gt_mel, mask_tensor):

        # assert delta_cam_info is not None
        head_mask = (mask_tensor >= 0.5)
        nonhead_mask = (mask_tensor < 0.5)

        coarse_data_dict = pred_dict["coarse_dict"]
        loss_dict = self.calc_data_loss(iter_, coarse_data_dict, gt_rgb, gt_rect, gt_mel, head_mask, nonhead_mask)

        total_loss = 0.0
        for k in loss_dict:
            total_loss += loss_dict[k]

        #cam loss
        if delta_cam_info is not None:
            loss_dict.update(self.calc_cam_loss(delta_cam_info))
            total_loss += 0.001 * loss_dict["delta_eular"] + 0.001 * loss_dict["delta_tvec"]

        # code loss
        loss_dict.update(self.calc_code_loss(opt_code_dict))
        total_loss += 0.001 * loss_dict["iden_code"] + \
                      1.0 * loss_dict["expr_code"] + \
                      0.001 * loss_dict["appea_code"] #+ \

        loss_dict["total_loss"] = total_loss
        return loss_dict
