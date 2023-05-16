import os
import cv2
import json
import random
import argparse
import imageio
from tqdm import tqdm
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from NetWorks.Wav2NeRFNet import Wav2NeRFNet
from NetWorks.audio_network import AudioNet, AudioAttNet
from options import BaseOptions
from Utils.Wav2NeRFLossUtils import Wav2NeRFLossUtils
from Utils.RenderUtils import RenderUtils
from Utils import audio
from Utils.hparams import hparams


# fix seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


class Wav2NeRF(object):
    def __init__(self, args, model_path, model_file, save_root, gpu_id) -> None:
        super().__init__()
        self.model_path = model_path
        self.model_file = model_file

        self.device = torch.device("cuda:%d" % gpu_id)
        self.save_root = save_root
        self.opt_cam = True
        self.view_num = 45
        self.data_path = args.data_path


        self.syncnet_mel_step_size = 16

        if not args.istest:
            self.syncnet_T = 5
            self.build_info_train()
            test_dir = os.path.join(self.data_path, 'transforms_train.json')
            with open(os.path.join(test_dir)) as fp:
                self.meta = json.load(fp)

        else:
            self.syncnet_T = 1 # rendering one frame
            self.build_info_render()
            test_dir = os.path.join(self.data_path, 'transforms_val.json')
            with open(os.path.join(test_dir)) as fp:
                self.meta = json.load(fp)

        self.build_tool_funcs()

        self.wav = audio.load_wav(os.path.join(self.data_path, 'aud.wav'), hparams.sample_rate)
        self.aud_features = np.load(os.path.join(self.data_path, 'aud.npy'))
        self.auds_ = torch.Tensor(self.aud_features).to(self.device) # [843,16,29]


    def build_info_train(self):

        # Load pre-trained HeadNeRF
        check_dict = torch.load(os.path.join(self.model_path, self.model_file), map_location=torch.device("cpu"))
        para_dict = check_dict["para"]
        
        self.opt = BaseOptions(para_dict)
        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size
        
        if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        net = Wav2NeRFNet(self.opt, include_vd=True, hier_sampling=False)        
        self.load_checkpoint(net, check_dict["net"])
        self.net = net.to(self.device)


        # Load Audio Network (modified)
        self.AudNet = AudioNet(dim_aud=256, win_size=16).to(self.device)
        self.AudAttNet = AudioAttNet().to(self.device)

        ckpt = torch.load(os.path.join(self.model_path, 'dfrf_base.tar')) # ./TrainedModels/dfrf_base.tar
        AudNet_state = ckpt['network_audnet_state_dict']
        AudAttNet_state = ckpt['network_audattnet_state_dict']
        self.load_checkpoint(self.AudNet, AudNet_state)
        self.load_checkpoint(self.AudAttNet, AudAttNet_state)



    def build_info_render(self):

        check_dict = torch.load(os.path.join(self.model_path, self.model_file), map_location=torch.device("cpu"))
        # para_dict = check_dict["para"]

        self.opt = BaseOptions(None)
        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size

        if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        net = Wav2NeRFNet(self.opt, include_vd=True, hier_sampling=False)        
        net.load_state_dict(check_dict["wav2nerf"]) 
        self.net = net.to(self.device)
        self.net.eval()

        self.AudNet = AudioNet(dim_aud=256, win_size=16).to(self.device)
        self.AudAttNet = AudioAttNet().to(self.device)

        self.AudNet.load_state_dict(check_dict["audnet"])
        self.AudAttNet.load_state_dict(check_dict["audattnet"])
        self.AudNet.eval()
        self.AudAttNet.eval()

        self.iden_offset = check_dict["code"]["iden"].to(self.device)
        self.expr_offset = check_dict["code"]["expr"].to(self.device)
        self.appea_offset = check_dict["code"]["appea"].to(self.device)

        self.delta_EulurAngles = check_dict["cam"]["delta_eulur"].to(self.device)
        self.delta_Tvecs = check_dict["cam"]["delta_tvec"].to(self.device)


    def build_tool_funcs(self):
        if not args.istest:
            self.loss_utils = Wav2NeRFLossUtils(self.model_path, device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy.repeat(self.syncnet_T,1,1)
        self.uv = self.render_utils.ray_uv.repeat(self.syncnet_T,1,1)
    

    def crop_audio_window(self, spec, pos_frame, neg_frame):

        start_idx = int(80. * (pos_frame / float(hparams.fps)))
        out_pos = torch.from_numpy(spec[start_idx:start_idx+self.syncnet_mel_step_size, :]).unsqueeze(0)

        negs = [None] * neg_frame.shape[0]
        for i in range(neg_frame.shape[0]):
            start_idx = int(80. * (neg_frame[i] / float(hparams.fps)))
            out_neg = spec[start_idx:start_idx+self.syncnet_mel_step_size, :]
            negs[i] = torch.from_numpy(out_neg).unsqueeze(0)
        out_negs = torch.cat(negs, dim=0) # [7,16,80]
        return out_pos, out_negs


    def load_data(self, img_path, mask_path, para_3dmm_path):

        # Batch idx ----------------------------------------
        start = self.meta['frames'][0]['img_id']
        end = self.meta['frames'][-1]['img_id']
        perm = np.random.permutation(np.arange(start+self.syncnet_T//2, end-self.syncnet_T//2-1)) # margin for syncnet loss
        batch_idx = range(perm[0]-self.syncnet_T//2, perm[0]+self.syncnet_T//2+1) # continuous image for syncnet loss

        # Audio for SyncNet loss
        orig_mel = audio.melspectrogram(self.wav).T
        mel_pos, mel_neg = self.crop_audio_window(orig_mel.copy(), perm[0], perm[1:16]) 
        self.gt_mel = torch.FloatTensor(mel_pos).permute(0,2,1).to(self.device) 
        self.neg_mel = torch.FloatTensor(mel_neg).permute(0,2,1).to(self.device)
        self.mels = torch.cat([self.gt_mel, self.neg_mel], dim=0)

        # Data per batch
        self.aud = [None] * self.syncnet_T
        self.img_tensor = [None] * self.syncnet_T
        self.mask_tensor = [None] * self.syncnet_T
        self.bg_tensor = [None] * self.syncnet_T
        self.base_c2w_Rmat = [None] * self.syncnet_T
        self.base_c2w_Tvec = [None] * self.syncnet_T
        self.temp_inv_inmat = [None] * self.syncnet_T
        self.rects = [None] * self.syncnet_T

        self.base_iden = [None]*self.syncnet_T
        self.base_expr = [None]*self.syncnet_T
        self.base_text = [None]*self.syncnet_T
        self.base_illu = [None]*self.syncnet_T

        for b_idx, idx in enumerate(batch_idx):

            image_file = str(idx) + '.png'

            # xywh of face ----------------------------------------
            rect = self.meta['frames'][idx]['face_rect'] # [4]
            self.rects[b_idx] = torch.Tensor(rect).unsqueeze(0).to(self.device)

            # audio smoothing ----------------------------------------
            smo_size = 8
            smo_half_win = int(smo_size / 2)
            left_i = idx - smo_half_win
            right_i = idx + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > end:
                pad_right = right_i-end
                right_i = end
            auds_win = self.auds_[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
            auds_win = self.AudNet(auds_win)
            self.aud_i = self.AudAttNet(auds_win).unsqueeze(0)
            self.aud[b_idx] = self.aud_i

            # Image & Mask ----------------------------------------
            img_size = (self.pred_img_size, self.pred_img_size)
            img_path_ = os.path.join(img_path, image_file)
            mask_path_ = os.path.join(mask_path, image_file[:-4]+'_mask'+image_file[-4:])

            img = cv2.imread(img_path_)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.0
            
            gt_img_size = img.shape[0]
            if gt_img_size != self.pred_img_size:
                img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            
            mask_img = cv2.imread(mask_path_, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            if mask_img.shape[0] != self.pred_img_size:
                mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            
            self.img_tensor_i = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
            self.mask_tensor_i = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
            self.img_tensor[b_idx] = self.img_tensor_i
            self.mask_tensor[b_idx] = self.mask_tensor_i

            # Background from image ----------------------------------------
            bg_path = os.path.join(self.data_path, "torso_imgs", image_file)
            bg = cv2.imread(bg_path)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg_tensor = bg.astype(np.float32)/255.0 # [512,512,3]
            self.bg_tensor_i = (torch.from_numpy(bg_tensor).permute(2, 0, 1)).unsqueeze(0).to(self.device)
            self.bg_tensor[b_idx] = self.bg_tensor_i


            # 3DMM ----------------------------------------
            # load init codes from the results generated by solving 3DMM rendering opt.
            para_3dmm_path_ = os.path.join(para_3dmm_path, image_file[:-4]+'_nl3dmm.pkl')
            with open(para_3dmm_path_, "rb") as f: nl3dmm_para_dict = pkl.load(f)

            base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
            base_iden = base_code[:, :self.opt.iden_code_dims]
            base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
            base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                                + self.opt.expr_code_dims + self.opt.text_code_dims]
            base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]

            self.base_iden[b_idx] = base_iden
            self.base_expr[b_idx] = base_expr
            self.base_text[b_idx] = base_text
            self.base_illu[b_idx] = base_illu

            # rotation & translation
            self.base_c2w_Rmat_i = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
            self.base_c2w_Tvec_i = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
            self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
            self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
            temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
            temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)
            
            temp_inv_inmat = torch.zeros_like(temp_inmat)
            temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
            temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
            temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
            temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
            temp_inv_inmat[:, 2, 2] = 1.0

            self.temp_inmat = temp_inmat
            self.temp_inv_inmat_i = temp_inv_inmat

            self.base_c2w_Rmat[b_idx] = self.base_c2w_Rmat_i
            self.base_c2w_Tvec[b_idx] = self.base_c2w_Tvec_i
            self.temp_inv_inmat[b_idx] = self.temp_inv_inmat_i

        # concat
        self.rects = torch.cat(self.rects, dim=0)
        self.aud = torch.cat(self.aud, dim=0) # [5,256]
        self.img_tensor = torch.cat(self.img_tensor, dim=0) # [5,3,512,512]
        self.mask_tensor = torch.cat(self.mask_tensor, dim=0) # [5,1,512,512]
        self.bg_tensor = torch.cat(self.bg_tensor, dim=0) # [5,3,512,512]

        self.base_iden = torch.cat(self.base_iden, dim=0)
        self.base_expr = torch.cat(self.base_expr, dim=0)
        self.base_text = torch.cat(self.base_text, dim=0)
        self.base_illu = torch.cat(self.base_illu, dim=0)
        self.base_c2w_Rmat = torch.cat(self.base_c2w_Rmat, dim=0) # [5,3,3]
        self.base_c2w_Tvec = torch.cat(self.base_c2w_Tvec, dim=0) # [5,3,1]
        self.temp_inv_inmat = torch.cat(self.temp_inv_inmat, dim=0) # [5,3,3]

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device)
        }
        


    def load_data_test(self, iter_, mask_path, para_3dmm_path):


        start = self.meta['frames'][0]['img_id']
        end = self.meta['frames'][-1]['img_id']

        smo_size = 8
        smo_half_win = int(smo_size / 2)
        left_i = iter_ - smo_half_win
        right_i = iter_ + smo_half_win
        pad_left, pad_right = 0, 0
        if left_i < 0:
            pad_left = -left_i
            left_i = 0
        if right_i > end:
            pad_right = right_i-end
            right_i = end
        auds_win = self.auds_[left_i:right_i]
        if pad_left > 0:
            auds_win = torch.cat(
                (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
        if pad_right > 0:
            auds_win = torch.cat(
                (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
        auds_win = self.AudNet(auds_win)
        self.aud = self.AudAttNet(auds_win)

        img_size = (self.pred_img_size, self.pred_img_size)
        mask_path = os.path.join(mask_path, str(iter_)+'_mask.png')        
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        self.mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        
        image_file = str(iter_) + '.png'
        bg_path = os.path.join(self.data_path, "torso_imgs", image_file)
        bg = cv2.imread(bg_path)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg_tensor = bg.astype(np.float32)/255.0 # [512,512,3]
        self.bg_tensor = (torch.from_numpy(bg_tensor).permute(2, 0, 1)).unsqueeze(0).to(self.device)

       # load init codes from the results generated by solving 3DMM rendering opt.
        para_3dmm_path = os.path.join(para_3dmm_path, str(iter_)+'_nl3dmm.pkl')
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
        
        self.base_iden = base_code[:, :self.opt.iden_code_dims]
        self.base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        self.base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        self.base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        gt_img_size=512
        temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device), # [1,3,3]
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device), # [1,3,1]
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device) # [1,3,3]
        }


    @staticmethod
    def eulurangle2Rmat(angles):
        batch_size = angles.size(0)
        
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()
        
        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx
        
        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz
        
        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res
    
    
    def build_code_and_cam(self):
        
        # code
        shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset], dim=-1)
        appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset
        
        opt_code_dict = {
            "bg":None,
            "iden":self.iden_offset,
            "expr":self.expr_offset,
            "appea":self.appea_offset
        }
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }

        #cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles, 
                "delta_tvec": self.delta_Tvecs
            }

            batch_delta_Rmats = self.eulurangle2Rmat(self.delta_EulurAngles).repeat(self.syncnet_T,1,1)
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = batch_delta_Rmats.bmm(base_Tvecs) + self.delta_Tvecs
            
            batch_inv_inmat = self.cam_info["batch_inv_inmats"]    
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat
            }
            
        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info


        return code_info, opt_code_dict, batch_cam_info, delta_cam_info
    
    
    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True


    def train(self,img_path, mask_path, para_3dmm_path, save_root):


        self.delta_EulurAngles = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.delta_Tvecs = torch.zeros((1, 3, 1), dtype=torch.float32).to(self.device)

        self.iden_offset = torch.zeros((1, 100), dtype=torch.float32).to(self.device)
        self.expr_offset = torch.zeros((1, 79), dtype=torch.float32).to(self.device)
        self.appea_offset = torch.zeros((1, 127), dtype=torch.float32).to(self.device)


        self.enable_gradient([*self.net.parameters(), *self.AudNet.parameters(), *self.AudAttNet.parameters()])
        if self.opt_cam:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset, self.delta_EulurAngles, self.delta_Tvecs]
            )
        else:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset]
            )

        init_learn_rate = 0.01
        step_decay = 100000 
        iter_num = 100000
        
        params_group = [
            {'params': [self.iden_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.expr_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.appea_offset], 'lr': init_learn_rate * 1.0},
            {'params': [*self.net.parameters()], 'lr': init_learn_rate * 0.01},
            {'params': [*self.AudNet.parameters()], 'lr': 5e-4},
            {'params': [*self.AudAttNet.parameters()], 'lr': 5e-4},
        ]
        if self.opt_cam:
            params_group += [
                {'params': [self.delta_EulurAngles], 'lr': init_learn_rate * 0.1},
                {'params': [self.delta_Tvecs], 'lr': init_learn_rate * 0.1},
            ]

        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        lr_func = lambda epoch: 0.1 ** (epoch / step_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        

        head_loss = []
        nonhaed_loss = []
        dwt_loss = []
        sync_loss = []
        vgg = []
        total_loss = []
        fig, axs = plt.subplots(2,3)

        for iter_ in range(iter_num):
            with torch.set_grad_enabled(True):

                self.load_data(img_path, mask_path, para_3dmm_path)
                code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam()

                # forward
                pred_dict = self.net("test", self.xy, self.uv, self.aud, self.bg_tensor, **code_info, **cam_info)

                # calculate loss
                batch_loss_dict = self.loss_utils.calc_total_loss(iter_,
                    delta_cam_info=delta_cam_info, opt_code_dict=opt_code_dict, pred_dict=pred_dict, 
                    gt_rgb=self.img_tensor, gt_rect=self.rects, gt_mel=self.mels, mask_tensor=self.mask_tensor)

                head_loss.append(batch_loss_dict["head_loss"].detach().cpu().numpy())
                nonhaed_loss.append(batch_loss_dict["nonhaed_loss"].detach().cpu().numpy())
                dwt_loss.append(batch_loss_dict["dwt_loss"].detach().cpu().numpy())
                vgg.append(batch_loss_dict["vgg"].detach().cpu().numpy())
                total_loss.append(batch_loss_dict["total_loss"].detach().cpu().numpy())

                if iter_ < 40000:
                    sync_loss.append(0)
                else:
                    sync_loss.append(batch_loss_dict["sync_loss"].detach().cpu().numpy())
                
                if iter_ % 500 == 0:
                    print(f"[{iter_}/{iter_num}] Head: {batch_loss_dict['head_loss']:.4f} / Nonhead: {batch_loss_dict['nonhaed_loss']:.8f} / DWT: {batch_loss_dict['dwt_loss']:.4f} / Sync: {batch_loss_dict['sync_loss']:.4f} / Total: {batch_loss_dict['total_loss']:.4f}")


            self.opt_code_dict = opt_code_dict
            self.delta_cam_info = delta_cam_info

            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()
            scheduler.step()   
            
            if iter_> 0 and iter_ % 10000 == 0:
                print("==== Saved Model ====")
                self.save_checkpoint(save_root, iter_)

                print("==== Saved Generated Image ====")
                coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
                coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                cv2.imwrite("%s/img_%d.png" % (save_root, iter_), coarse_fg_rgb[:, :, ::-1])

                print("==== Saved Loss Curve====")
                axs[0,0].plot(np.arange(iter_+1), head_loss, linewidth=0.7)
                axs[0,1].plot(np.arange(iter_+1), nonhaed_loss, linewidth=0.7)
                axs[0,2].plot(np.arange(iter_+1), dwt_loss, linewidth=0.7)
                axs[1,0].plot(np.arange(iter_+1), sync_loss, linewidth=0.7)
                axs[1,1].plot(np.arange(iter_+1), vgg, linewidth=0.7)
                axs[1,2].plot(np.arange(iter_+1), total_loss, linewidth=0.7)
                plt.savefig('%s/loss_curve_%d.png' % (save_root, iter_))

        # to save
        self.opt_code_dict = opt_code_dict
        self.delta_cam_info = delta_cam_info


    def perform_rendering(self, img_path, mask_path, para_3dmm_path, save_root):
        
        import json

        test_len = len(self.meta['frames']) 
        start = self.meta['frames'][0]['img_id']
        end = self.meta['frames'][-1]['img_id']
        res_img_list = []
        for i, iter_ in enumerate(range(start, end)):
            if i%50==0:
                print(i, "/", test_len)

            self.load_data_test(iter_, mask_path, para_3dmm_path)
            code_info, _, cam_info, _ = self.build_code_and_cam()
            pred_dict = self.net("test", self.xy, self.uv, self.aud, self.bg_tensor, **code_info, **cam_info)

            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            cv2.imwrite("./%s/%d.png" % (save_root, iter_), coarse_fg_rgb[:, :, ::-1])
            res_img_list.append(coarse_fg_rgb)

        NVRes_save_path = "%s/results.mp4" % (save_root)
        imageio.mimwrite(NVRes_save_path, res_img_list, fps=25, quality=8)
        print("Rendering Done")


    def save_checkpoint(self, save_root, iter):
        
        model_dict = {
            "code": self.opt_code_dict,
            "cam": self.delta_cam_info,
            "wav2nerf": self.net.state_dict(),
            "audnet": self.AudNet.state_dict(),
            "audattnet": self.AudAttNet.state_dict()
        }
        torch.save(model_dict, "%s/Wav2NeRF_%s.pth" % (save_root, str(iter)))


    def load_checkpoint(self, ours, checkpoint:dict)->None:
        
        state_dict = checkpoint
        model_state_dict = ours.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print("Skip loading pararmeter")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print("Dropping parameter")
                is_changed = True


    def run(self, img_path, mask_path, para_3dmm_path, save_root):
        
        self.train(img_path, mask_path, para_3dmm_path, save_root)
        self.save_checkpoint(save_root, "final")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='a framework for talking head generation')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--istest", type=bool, default=False)
    
    args = parser.parse_args()

    model_path = args.model_path
    model_file = args.model_file
    data_path = args.data_path
    save_root = args.save_root
    img_path = os.path.join(data_path, 'ori_imgs')
    mask_path = os.path.join(data_path, 'ori_imgs')
    para_3dmm_path = os.path.join(data_path, 'ori_imgs')
    istest = args.istest
    
    tt = Wav2NeRF(args, model_path, model_file, save_root, gpu_id=0)
    if not istest:
        tt.run(img_path, mask_path, para_3dmm_path, save_root)
    else:
        tt.perform_rendering(img_path, mask_path, para_3dmm_path, save_root)
