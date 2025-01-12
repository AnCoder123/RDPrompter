import torch
import torch.nn as nn
import torch.nn.functional as F
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter
import numpy as np
import cv2


from .AffinityLearner import AffinityLearner
from .CSAttention import CSAttention
from .SparsePromptLearner import SparsePromptLearner


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class BaseExtendSam(nn.Module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h','vit_t'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)
        self.DRconv1_L = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1)
        self.DRconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        transformer_dim=256
        vit_dim=256
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2))

        self.LRweight1=torch.nn.Parameter(torch.rand(256))
        self.LRweight2=torch.nn.Parameter(torch.rand(256))
        # self.LRweight3=torch.nn.Parameter(torch.rand(256))
        # self.LRweight4=torch.nn.Parameter(torch.rand(256))

        self.CSAttention=CSAttention()
        self.AffinityLearner = AffinityLearner(2,1)
        self.SparsePromptLearner=SparsePromptLearner().cuda()


    # Masked Average Pooling
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(3, 4)) \
            / (mask[None, ...].sum(dim=(3, 4)) + 1e-5) # 1 x C
        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def getPrototype_learnable(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        # n_ways=1 n_shots=5
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        for way in fg_fts:
            c=torch.var(way)
        fg_prototypes_mean = [sum(way) / n_shots for way in fg_fts]
        if fg_fts[0].shape[0]==1:
            fg_prototypes_var=[torch.zeros_like(way).cuda() for way in fg_fts]
        else:
            fg_prototypes_var=[torch.var(way,dim=0) for way in fg_fts]
        bg_prototype_mean = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        bg_prototype_var = [torch.var(way,dim=0) for way in bg_fts]


        return fg_prototypes_mean,fg_prototypes_var, bg_prototype_mean,bg_prototype_var


    def vistensor(self,finalpred,name):
        demo=finalpred
        array = demo.detach().cpu().numpy()
        image = np.uint8(array * 255)
        color_map = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        cv2.imwrite(name, color_map)


    def vistensor_img(self,finalpred,name):
        demo=finalpred
        array = demo.detach().cpu().numpy()
        image = np.uint8(array*255)
        color_map=image.transpose(1, 2, 0)
        # color_map = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        cv2.imwrite(name, color_map)
        

    def calDist_bg(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler

        
        return dist

    def calDist_fg(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        # dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) 
        # normalized_tensor = (dist - dist.min()) / (dist.max() - dist.min())* scaler
        # return normalized_tensor
        return dist
        

    def calDist_bg(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        # dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        # dd=prototype[..., None, None]
        # dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) 
        # normalized_tensor = (1-(dist - dist.min()) / (dist.max() - dist.min()))* scaler
        # return normalized_tensor
    
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler*(-1)
        return dist
    
    def getMaskFeatures(self, fts, mask):
        """
        Extract mask feature
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
    
        masked_fts=fts * mask#[None, ...]
        return masked_fts


    def forward(self, img,supportMasks,queryLabels):
        img=img.squeeze(0)
        x,interm_embeddings = self.img_adapter(img)
        
        # orix=x[5,:,:,:].unsqueeze(0)  # 5fots
        orix=x[1,:,:,:].unsqueeze(0)  # 1fots 

        early_x = self.DRconv1_L(interm_embeddings[0].permute(0, 3, 1, 2))
        center_x = self.DRconv1_L(interm_embeddings[2].permute(0, 3, 1, 2))
        x= self.DRconv2(x)
        x= torch.cat([x,early_x,center_x],dim=1)


        supportMasks_fg=torch.transpose(supportMasks,0,1)
        supportMasks_bg=torch.logical_not(supportMasks_fg)
        # supportImgEmbedding=x[0:5,:,:,:]  # 5 fots
        supportImgEmbedding=x[0:1,:,:,:]  # 1 fots
        supp_fg_fts=self.getFeatures(supportImgEmbedding,supportMasks_fg)
        supp_bg_fts=self.getFeatures(supportImgEmbedding,supportMasks_bg)
        fg_prototypes_mean,fg_prototypes_var, bg_prototype_mean,bg_prototype_var = self.getPrototype_learnable(supp_fg_fts, supp_bg_fts)
        eps1=torch.randn_like(fg_prototypes_var[0]) 
        # eps2=torch.randn_like(fg_prototypes_var[0])
        fg_prototypes=torch.mul(self.LRweight1, fg_prototypes_mean[0])+eps1*fg_prototypes_var[0]*self.LRweight2

        x=x[1,:,:,:].unsqueeze(0)   # 1 fots
        dist_fg = self.calDist_fg(x, fg_prototypes[0])
 

        supportMasks_fg=torch.transpose(supportMasks,0,1).float()
        supportMasks_fg = F.interpolate(supportMasks_fg, size=(64, 64), mode='bilinear', align_corners=False)
        supp_mask_fts=self.getMaskFeatures(supportImgEmbedding,supportMasks_fg)
        dist_fg2 = self.CSAttention(x.permute(0,3,2,1), supp_mask_fts.permute(0,3,2,1))


        dist=[dist_fg,dist_fg2.squeeze(3)]
        distTensor = torch.stack(dist, dim=0)  # N x (1 + Wa) x H' x W'
        distTensor = torch.transpose(distTensor,0,1)

        DTensor=self.AffinityLearner(distTensor)
        DTensor_Resize=F.interpolate(DTensor, size=(256,256), mode='bilinear')
        Dmask=DTensor_Resize[:,0,:,:]
        masks = Dmask
        masks = torch.unsqueeze(masks, 0)
        _, dense_embeddings = self.prompt_adapter(points=None,boxes=None,masks=masks)


        ImgEmbedding=x.flatten(2).permute(2,0,1)  #1*256*64*64
        GlobalSim=dist_fg #1*64*64
        localSim=dist_fg2.squeeze(3) #1*64*64
        sparse_embeddingsLearnable=self.SparsePromptLearner(ImgEmbedding,GlobalSim,localSim)


        # attn_sim
        # attn_sim1=GlobalSim.squeeze(0)
        # attn_sim1 = (attn_sim1 - attn_sim1.mean()) / torch.std(attn_sim1)
        # attn_sim1=attn_sim1.unsqueeze(0).unsqueeze(0)
        # attn_sim1 = attn_sim1.sigmoid_().unsqueeze(0).flatten(3)
        # attn_sim2=localSim.squeeze(0)
        # attn_sim2 = (attn_sim2 - attn_sim2.mean()) / torch.std(attn_sim2)
        # attn_sim2=attn_sim2.unsqueeze(0).unsqueeze(0)
        # attn_sim2 = attn_sim2.sigmoid_().unsqueeze(0).flatten(3)
        # attn_sim=torch.cat([attn_sim1,attn_sim2],dim=0)
        attn_sim=None
        
        
        # Mask Encoder
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=orix,
            multi_image_embeddings=x, 
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddingsLearnable,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attn_sim=attn_sim
        )
        
        return low_res_masks, iou_predictions


class SemanticSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, class_num=2, model_type='vit_b'):
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)
