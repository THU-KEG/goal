from src.modeling.transformer.transformer import DecoderEmbeddings, TransformerDecoderLayer, TransformerDecoder, generate_square_subsequent_mask
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Tuple
from einops import repeat




class VisualTransformer(nn.Module):
    def __init__(self, tokenizer_dir: str,
                 max_dec_length: int=300,
                 d_model: int=768,
                 n_head: int=8,
                 dim_feedforward: int=2048,
                 activation: str='relu',
                 num_decoder_layers: int=6,
                 normalize_before = False,
                 layer_norm_eps:float = 1e-5,
                 dropout:float = 0.1,
                 return_intermediate = False,
                ) -> None:
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.d_model = d_model
        self.vocab_size = self.tokenizer.vocab_size
        self.max_dec_length = max_dec_length
        
        # build model
        self.embeddings = DecoderEmbeddings(self.tokenizer.vocab_size, d_model,
                                            self.tokenizer.pad_token_id, max_dec_length, layer_norm_eps, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                               return_intermediate=return_intermediate)
        
        # self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model), nn.ReLU, nn.Linear(d_model, self.vocab_size, bias=False)])
        self.out_proj = nn.Linear(d_model, self.vocab_size, bias=False)
        
        self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, interleaved_embeds, interleaved_types, interleaved_mask, target_input_ids, target_input_mask):
        """ Calculate the decoder outputs given encoder memory, target (input_ids), and target mask.
          Args:
            target_ids
        """
        b, l, d = interleaved_embeds.shape # (B, 2TL, d)
        b, t, s = target_input_ids.shape # (B, T, S)

        memory_embed = interleaved_embeds.permute(1, 0, 2)
        memory_key_mask = torch.ones(b, s) < 0
            
        # target = self.embeddings(target_input_ids[:, :-1]).permute(1, 0, 2)
        target = self.embeddings(target_input_ids.view(b, -1)).permute(1, 0, 2)
        target_key_mask = target_input_mask.view(b, -1) <= 0

        # # Build cross-attention mask that input_ids attend to the interleaved context
        # interleaved_mask = torch.zeros(b, t*s, l).fill_(torch.finfo.min)
        # for i in  range(t):
        #     # the current i th textual sequence attent on previous i-1 pairs of text-visual sequences and i th visual sequence
        #     interleaved_mask[:, i:i+i*s, :] = torch.where(repeat(interleaved_types<=2*i+2, 'b l -> b s l', s=s), 0.0, torch.finfo.min)

        
        query_pos = self.embeddings.position_embeddings.weight.unsqueeze(1)
        # query_embed = query_embed.repeat(1, bsz, 1)
        query_pos = query_pos.expand(query_pos.shape[0], b, query_pos.shape[2])
        
        # hs = self.decoder(target, memory_embed, memory_key_padding_mask=memory_mask, tgt_key_padding_mask=target_input_mask[:, :-1],
        hs = self.decoder(target, memory_embed, memory_key_padding_mask=memory_key_mask, tgt_key_padding_mask=target_key_mask,
                          query_pos=query_pos, memory_mask= interleaved_mask,
                          tgt_mask=generate_square_subsequent_mask(len(target)).to(target.device))
        logits = self.out_proj(hs.permute(1, 0, 2))
        
        # Compute loss
        # loss = self.loss_func(logits.permute(0,2,1), target_input_ids[:,1:])
        # loss = self.loss_func(logits.permute(0,2,1)[:, :-1, :], target_input_ids[:, 1:])
        loss = self.loss(logits[:, :-1, :].contiguous(), target_input_ids[:, 1:].contiguous(), target_input_mask[:,1:])
        
        return logits, loss
    

    
    def generate(self, memory_embed, memory_mask, pos_embed=None, target_input_ids=None, target_input_mask=None):
        """ Generate the target sequence given encoder memory.
              Calculate the loss if the target_ids and target_mask are given.
          Return:
            caption_ids [B, max_dec_len]: the targets padded with start/end tokens.
            caption_logits [B, max_dec_len]: the targets padded with end token logits.
        """
        # if self.training:
        #     self.eval()
        bsz, n_seq, d_emb = memory_embed.shape
        caption_ids, caption_mask = self.prepare_input(self.tokenizer, self.max_dec_length, batch_size=bsz)
        caption_ids = caption_ids.to(memory_embed.device)
        caption_mask = caption_mask.to(memory_embed.device)
        # caption_logits = torch.zeros(size=(bsz, self.max_dec_length+1, self.tokenizer.vocab_size), device=memory_embed.device)
        caption_logits = torch.zeros(size=(bsz, self.max_dec_length, self.tokenizer.vocab_size), device=memory_embed.device)

        bsz_ends = torch.zeros(size=(bsz,), dtype=torch.int32)
        i = 0
        while not all(bsz_ends) and i < self.max_dec_length-1:
            logits, _ = self.forward(memory_embed, memory_mask, caption_ids, caption_mask, pos_embed)
            logits = logits[:, i, :]
            pred_ids = torch.argmax(logits, axis=-1) # (B, )

            caption_ids[:, i+1] = pred_ids
            # caption_mask[:, i+1] = False
            caption_mask[:, i+1] = 1.0
            caption_logits[:, i, :] = logits

            _ends = (pred_ids == self.tokenizer.sep_token_id).detach().cpu().int()
            bsz_ends = torch.where(bsz_ends!=0, bsz_ends, _ends*(i+1))

            i += 1
        
        target_captions = list(map(lambda x: self.tokenizer.decode(x[1:-1], skip_special_tokens=True), caption_ids))

        loss = None
        if target_input_ids is not None:
            loss = self.loss(caption_logits[:, 1:, :].contiguous(), target_input_ids[:, :-1].contiguous())

        return (caption_ids, caption_logits, bsz_ends, target_captions, loss)
    
    
    def loss(self, logits, labels, mask=None) -> torch.tensor:
        """ Computes Causal Language Modeling (CLM) loss value according to the loss function.
          Args:
            logits:
            labels: 
        """
        bsz, seq_len, v = logits.shape
        # loss = self.loss_func(logits.view(-1, v), labels.view(-1))
        loss = F.cross_entropy(logits.view(-1, v), labels.view(-1), ignore_index=0)
        return loss
    
    
    @classmethod
    def prepare_input(self, tokenizer, max_dec_length, batch_tokens=None, batch_size=None) -> Tuple:
        """ Prepare the inputs to decoder. Generate padded targets if the batch_tokens is not None, else generate the start tokens for infenrence.
        """
        assert batch_tokens or batch_size
        
        if batch_tokens is not None:
            batch_dec = tokenizer.batch_encode_plus(
                            batch_tokens,
                            max_length=max_dec_length,
                            padding='max_length',
                            return_tensors="pt",
                            truncation=True
                        )
            target_caption_ids = batch_dec.input_ids  # (B, L)
            target_caption_mask = batch_dec.attention_mask  # (B, L)
            # gold_output_ids = target_caption_ids[1:]
        else:
            start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
            end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
            target_caption_ids = torch.zeros((batch_size, max_dec_length), dtype=torch.long)
            # target_caption_mask = torch.ones((batch_size, max_dec_length), dtype=torch.bool)
            target_caption_mask = torch.zeros((batch_size, max_dec_length), dtype=torch.long)

            target_caption_ids[:, 0] = start_token
            # target_caption_mask[:, 0] = False
            target_caption_mask[:, 0] = 1.0
            # gold_output_ids = None
            
        return target_caption_ids, target_caption_mask
        # return target_caption_ids, target_caption_mask, gold_output_ids
            
    
    @classmethod
    def process_output(self, pred_ids, pred_lens, tokenizer):
        """ Get the output caption sentences given batch prediction indices.
        """
        result = []
        for i in range(len(pred_ids)):
            dec_sent = tokenizer.decode(pred_ids[i][:pred_lens[i]], skip_special_tokens=True)
            result.append(dec_sent)
        return result