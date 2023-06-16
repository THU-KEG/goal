import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from einops import rearrange, reduce, repeat
from horovod import torch as hvd
from src.modeling.timesformer.vit import VisionTransformer
from src.modeling.xbert import (BertEmbeddings, BertEncoder, BertForMaskedLM,
                                BertLMPredictionHead, BertModel, BertPooler,
                                BertPreTrainedModel, BertPreTrainingHeads)
from src.utils.basic_utils import load_json, load_jsonl, save_frames_grid
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from src.modeling.fla_transformer.visual_transformer import VisualTransformer
from src.modeling.fla_gpt2.visual_gpt2 import VisualGPT2
from src.modeling.layers.resampler import PerceiverResampler


class FlaBaseModel(nn.Module):
    def __init__(self, cfg, text_enc_cfg=None, visual_enc_cfg=None, text_dec_cfg=None, input_format='RGB', temp=0.07, freeze_v_bachbone=False, freeze_t_backbone=False):
        super().__init__()
        
        self.temp = nn.Parameter(torch.ones([]) * temp)   
        # FIXME make them configurable
        embed_dim = 256
        vision_width = 768
        text_width = text_enc_cfg.hidden_size

        self.n_visual_latens = cfg['n_visual_latents']
        self.n_text_latens = cfg['n_text_latents']

        # Encoders
        self.visual_encoder = VisionTransformer(visual_enc_cfg['img_size'], visual_enc_cfg['patch_size'], attention_type='space_only')
        self.text_encoder = BertForMaskedLM.from_pretrained(text_enc_cfg.bert_pretrained_path, config=BertConfig(**text_enc_cfg))

        # Decoder
        if text_dec_cfg['model'] == 'transformer':
            self.decoder = VisualTransformer(
                text_dec_cfg['tokenizer_dir'],
                text_dec_cfg['max_dec_length'],
                text_dec_cfg['d_model'],
                text_dec_cfg['n_head'],
                text_dec_cfg['dim_feedforward'],
                text_dec_cfg['activation'],
                text_dec_cfg['n_dec_layers'],
                text_dec_cfg['normalize_before'],
                text_dec_cfg['layer_norm_eps'],
                text_dec_cfg['dropout'],
                text_dec_cfg['return_intermediate']
            )
        elif text_dec_cfg['model'] == 'gpt2':
            self.decoder = VisualGPT2(
                text_dec_cfg['config_file'],
                text_dec_cfg['pretrained_dir'],
                text_dec_cfg['freeze_pretrained']
            )
        else:
            raise BaseException("Not supported decoder model %s" % text_dec_cfg['model'])
        

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itc_token_type = cfg.itc_token_type
        self.itm_head = nn.Linear(text_width, 2)

        # Resampler
        self.visual_resampler = PerceiverResampler(
            dim=embed_dim,
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=cfg['n_visual_latents'],
            num_time_embeds=1024
        )

        self.text_resampler = PerceiverResampler(
            dim=embed_dim,
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=cfg['n_text_latents'],
            num_time_embeds=1024
        )

        self.freeze_backbones(freeze_textual=freeze_t_backbone, freeze_visual=freeze_v_bachbone)

    def load_separate_ckpt(self, visual_enc_weights=None, text_enc_weights=None):
        if visual_enc_weights:
            self.visual_encoder.load_state_dict(visual_enc_weights)


    def freeze_backbones(self, freeze_textual=False, freeze_visual=False):
        if freeze_textual:
            for n, p in self.text_encoder.named_parameters():
                p.requires_grad = False
        if freeze_visual:
            for n, p in self.visual_encoder.named_parameters():
                p.requires_grad = False


class FlaForPretrain(FlaBaseModel):
    def __init__(self, cfg, text_enc_cfg, visual_enc_cfg, text_dec_cfg, input_format='RGB'):
        super(FlaForPretrain, self).__init__(cfg, text_enc_cfg, visual_enc_cfg=visual_enc_cfg, input_format=input_format)

        self.use_mask_prob = 0


    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs'] # (B, T, C, H, W)
        text_input_ids = batch['text_input_ids'] # (B, T, S)
        text_input_mask = batch['text_input_mask']
        # target_input_ids = batch['target_input_ids'] # (B, T, S)
        # target_input_mask = batch['target_input_mask']
        # target_label_ids = batch['target_label_ids']

        device = visual_inputs.device
        b, t, c, h, w = visual_inputs.shape
        b, t, s = text_input_ids.shape

        # forward image and text features
        # feats are normalized embeds
        visual_embeds = self._forward_visual_embeds(visual_inputs) # (B, T, L+1, d)

        # we compute normalized feats for unmasked visual inputs only, used for ITC
        visual_feat = F.normalize(self.vision_proj(visual_embeds[:,:,0, :]),dim=-1)
        visual_atts = torch.ones(visual_embeds.size()[:-1],dtype=torch.long).to(device)
        
        # text embeddings and features
        text_embeds, text_feat = self._forward_text_feats(text_input_ids, text_input_mask) # (B, T, S+1, d)
        text_atts = text_input_mask

        # resample to fixed number of tokens
        visual_perceiv = self.visual_resampler(visual_embeds) # (B, Q, d)
        text_perceiv = self.text_resampler(text_embeds, text_atts) # (B, Q, d)
        q, d = visual_perceiv.shape[-2:]

        # build interleaved sequence for generative loss
        visual_perceiv = visual_perceiv.split(1, dim=1)
        text_perceiv = text_perceiv.split(1, dim=1)
        assert len(visual_perceiv) == len(text_perceiv) == t

        interleaved_embeds = []
        interleaved_types = []
        for i in range(t):
            interleaved_embeds.append(torch.cat([visual_perceiv[i], text_perceiv[i]], dim=1)) # (B, 2L, d)
            interleaved_types.append(torch.cat([torch.ones(b,q)*i, torch.ones(b,q)*(i+1)], dim=1))
        interleaved_embeds = torch.cat(interleaved_embeds, dim=1) # (B, 2TL, d)
        interleaved_types = torch.cat(interleaved_types, dim=1) # (B, 2TL)

        # Build cross-attention mask that input_ids attend to the interleaved context
        interleaved_mask = torch.zeros(b, t*s, 2*t*q).fill_(torch.finfo.min)
        for i in  range(t):
            # the current i th textual sequence attent on previous i-1 pairs of text-visual sequences and i th visual sequence
            interleaved_mask[:, i:i+i*s, :] = torch.where(repeat(interleaved_types<=2*i+2, 'b l -> b s l', s=s), 0.0, torch.finfo.min)


        # ========== (in-batch) ITC loss ==========
        t_visual_feat = visual_feat.view(-1, visual_feat.shape[-1])
        t_text_feat = text_feat.view(-1, visual_feat.shape[-1])

        gathered_video_feats = hvd.allgather(t_visual_feat) # (G*B*T, d)
        gathered_text_feats = hvd.allgather(t_visual_feat) # (G*B*T, d)

        assert self.itc_token_type == 'cls', 'Support CLS tokens for ITC only, find {}.'.format(self.itc_token_type)
        sim_v2t = t_visual_feat @ gathered_text_feats.t() / self.temp 
        sim_t2v = t_text_feat @ gathered_video_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_v2t 
        # allgather return the concatenated features in the order of local_rank()
        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * t * local_rank, b * t * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

        vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= VTM ==========

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos = self.compute_vtm(text_embeds=text_perceiv, 
                                                                                 video_embeds=visual_perceiv, 
                                                                                 sim_v2t=sim_v2t.clone(), # for hard mining
                                                                                 sim_t2v=sim_t2v.clone(), # for hard mining
                                                                                 return_encoder_out=True
                                                                                )

        # ========= LM ==========
        lm_logits, lm_loss, lm_states = self.decoder(interleaved_embeds, interleaved_types, interleaved_mask, text_input_ids, text_input_mask)

        return dict(
            itc_loss=vtc_loss,
            itm_scores=vtm_logits,  # (B, 2)
            itm_loss=vtm_loss,  # (1, )
            itm_labels=vtm_labels,  # (B, )
            lm_loss=lm_loss,
            lm_logits=lm_logits,
            lm_states=lm_states
        )


    def _forward_visual_embeds(self, visual_inputs):
        """ Encode T images into T sequences of tokens using Visual Encoder.
        """
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        visual_inputs = visual_inputs.transpose(1, 2)
        visual_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)

        return visual_embeds

    def _forward_text_feats(self, text_input_ids, text_input_mask):
        # text features
        b, t, s = text_input_ids.shape

        text_output = self.text_encoder.bert(text_input_ids.view(-1, s), 
                                             attention_mask=text_input_mask.view(-1, s),                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )

        text_embeds = text_output.last_hidden_state # b*t, s, d=768
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        text_embeds = text_embeds.view(b,t,s, -1)
        text_feat = text_feat.view(b,t,-1)

        return text_embeds, text_feat



    def compute_vtm(self, text_embeds, visual_embeds, sim_v2t, sim_t2v, return_encoder_out=False):
        """ Compute visusal-textual matching loss given resampled visual/textual tokens.
        """
        b, t, l, d = text_embeds.shape
        device = text_embeds.device

        t_visual_embeds = visual_embeds.view(-1, l, d)
        t_text_embeds = text_embeds.view(-1, l, d)

        # ====== positive pairs =======
        embedding_output_pos = torch.cat([t_text_embeds, t_visual_embeds], dim=1)
        att_mask = torch.ones(b*t, 2*l)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=att_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        # ====== negative pairs =======
        bs = t_text_embeds.shape[0] 

        local_rank = hvd.local_rank()
        b_start, b_end = bs * t * local_rank, bs * t * (local_rank + 1)

        with torch.no_grad():
            weights_i2t = sim_v2t[:,b_start:b_end]
            weights_t2i = sim_t2v[:,b_start:b_end]
   
            # never select self as negative
            weights_i2t.fill_diagonal_(-np.Inf)
            weights_t2i.fill_diagonal_(-np.Inf)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        t_visual_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            t_visual_embeds_neg.append(t_visual_embeds[neg_idx])
        t_visual_embeds_neg = torch.stack(t_visual_embeds_neg,dim=0)   

        # select a negative text for each image
        t_text_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            t_text_embeds_neg.append(t_text_embeds[neg_idx])

        t_text_embeds_neg = torch.stack(t_text_embeds_neg,dim=0)   

        t_text_embeds_all = torch.cat([t_text_embeds, t_text_embeds_neg],dim=0)     
        t_text_atts_all = torch.ones(b*t, 2*l)     

        t_visual_embeds_all = torch.cat([t_visual_embeds_neg,t_visual_embeds],dim=0)
        t_visual_atts_all = torch.ones(b*t, 2*l)

        attention_mask_all = torch.cat([t_text_atts_all, t_visual_atts_all], dim=1)
        embedding_output_all = torch.cat([t_text_embeds_all, t_visual_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vtm_logits = self.itm_head(vl_embeddings)            

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)

        vtm_loss = vtm_loss.view(b, t)
        vtm_logits = vtm_logits.view(b, t, -1)
        vtm_labels = vtm_labels.view(b, t)
        encoder_outputs_pos = encoder_outputs_pos.view(b, t, -1)

        if return_encoder_out:
            return vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos 
        else:
            return vtm_loss, vtm_logits, vtm_labels, None
        




class FlaForVideoTextGeneration(FlaBaseModel):
    """
    """
    def __init__(self, config, model_config, video_enc_cfg, dec_cfg, input_format='RGB'):
        super(FlaForVideoTextGeneration, self).__init__(
            model_config, input_format=input_format, video_enc_cfg=video_enc_cfg,
            freeze_v_bachbone=config.freeze_visual_backbone, freeze_t_backbone=config.freeze_textual_backbone)
        
        self.embed_dim = 768
        self.d_model = dec_cfg['d_model']
        self.multimodal_fuse_proj = nn.Linear(self.embed_dim, self.d_model)
        
        # self.position_embedding = build_position_encoding(self.d_model)
        self.position_embedding = nn.Embedding(video_enc_cfg['patch_size']**2+1+config.max_enc_txt_len,
                                               self.d_model, padding_idx=0)

        # Initialize resampler
        self.resampler = PerceiverResampler(
            dim=self.embed_dim,
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_time_embeds=1024
        )
        
        # Initialize decoder
        if dec_cfg['model'] == 'transformer':
            self.decoder = VisualTransformer(
                dec_cfg['tokenizer_dir'],
                dec_cfg['max_dec_length'],
                dec_cfg['d_model'],
                dec_cfg['n_head'],
                dec_cfg['dim_feedforward'],
                dec_cfg['activation'],
                dec_cfg['n_dec_layers'],
                dec_cfg['normalize_before'],
                dec_cfg['layer_norm_eps'],
                dec_cfg['dropout'],
                dec_cfg['return_intermediate']
            )
        elif dec_cfg['model'] == 'gpt2':
            self.decoder = VisualGPT2(
                dec_cfg['config_file'],
                dec_cfg['pretrained_dir'],
                dec_cfg['freeze_pretrained']
            )
        else:
            raise BaseException("Not supported decoder model %s" % dec_cfg['model'])
            
            
            

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']
        prompt_input_ids = batch['prompt_input_ids']
        prompt_input_mask = batch['prompt_input_mask']
        target_input_ids = batch['target_input_ids']
        target_input_mask = batch['target_input_mask']
        # target_label_ids = batch['target_label_ids']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # visual embeddings
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True) # (B, N_patch+1, d)
        # image_embeds = image_embeds.repeat(text_input_mask.shape[0], 1, 1)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # text embeddings
        text_output = self.text_encoder.bert(prompt_input_ids,
                                             attention_mask=prompt_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state # (B, L, d)
        # text_embeds = F.normalize(self.text_proj(text_embeds),dim=-1)     
        # text_embeds = self.text_proj(text_embeds)   
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        
        # decode
        memory = self.multimodal_fuse_proj(torch.cat([video_embeds, text_embeds], 1))
        # memory_mask = torch.cat([video_atts, prompt_input_mask], dim=1) <= 0 # (B, N+L)
        memory_mask = torch.cat([video_atts, prompt_input_mask], dim=1) # (B, N+L)
        pos_embed = self.position_embedding(torch.arange(memory.shape[1], device=memory.device).unsqueeze(0).expand(b, memory.shape[1]))
        # target_input_mask = target_input_mask <= 0
        
        # lm_logits, lm_loss = self.decoder(memory, memory_mask, target_input_ids, target_input_mask, target_label_ids, pos_embed=pos_embed)
        lm_logits, lm_loss, lm_states = self.decoder(memory, memory_mask, target_input_ids, target_input_mask, pos_embed=pos_embed)

        return dict(
            logits=lm_logits,
            loss=lm_loss,
        )
        


    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        prompt_input_ids = batch['prompt_input_ids']
        prompt_input_mask = batch['prompt_input_mask']

        # target_input_ids = batch['target_input_ids'][:, 1:] if 'target_input_ids' in batch else None
        target_input_ids = batch['target_input_ids'] if 'target_input_ids' in batch else None
        # target_input_mask = batch['target_input_mask']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # visual embeddings
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True) # (B, N_patch+1, d)
        # image_embeds = image_embeds.repeat(text_input_mask.shape[0], 1, 1)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # text embeddings
        text_output = self.text_encoder.bert(prompt_input_ids,
                                             attention_mask=prompt_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state # (B, L, d)
        # text_embeds = F.normalize(self.text_proj(text_embeds),dim=-1)     
        # text_embeds = self.text_proj(text_embeds)   
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        
        # decode
        memory = self.multimodal_fuse_proj(torch.cat([video_embeds, text_embeds], 1))
        # memory_mask = torch.cat([video_atts, prompt_input_mask], dim=1) <= 0 # (B, N+L)
        memory_mask = torch.cat([video_atts, prompt_input_mask], dim=1) # (B, N+L)
        pos_embed = self.position_embedding(torch.arange(memory.shape[1], device=memory.device).unsqueeze(0).expand(b, memory.shape[1]))
        
        # pred_ids, logits, pred_lens, pred_captions, loss = self.decoder.generate(memory, memory_mask, pos_embed=None, target_input_ids=target_input_ids)
        pred_ids, logits, pred_lens, pred_captions, loss = self.decoder.generate(memory, memory_mask, pos_embed=pos_embed, target_input_ids=target_input_ids)

        return dict(
            pred_ids=pred_ids,
            logits=logits,
            pred_lens=pred_lens,
            pred_captions=pred_captions,
            loss=loss
        )

    
    
