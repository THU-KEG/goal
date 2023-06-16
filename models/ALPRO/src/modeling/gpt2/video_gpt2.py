from src.modeling.gpt2.configuration_gpt2 import GPT2Config
from src.modeling.gpt2.modeling_gpt2 import GPT2LMHeadModel
from src.modeling.gpt2.tokenization_gpt2 import GPT2Tokenizer
from src.modeling.gpt2.utils_gpt2 import *
from src.utils.basic_utils import load_json
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple




class VideoGPT2(nn.Module):
    def __init__(self, config_file, pretrained_dir=None, freeze_pretrained=True,
                 return_intermediate = False,
                ) -> None:
        super().__init__()
        
        gpt2_config = GPT2Config.from_pretrained(config_file)
        gpt2_config.add_cross_attention = True

        # self.tokenizer = GPT2Tokenizer(vocab_file, merges_file)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.d_model = gpt2_config.n_embd
        self.vocab_size = self.tokenizer.vocab_size
        self.max_dec_length = gpt2_config.n_ctx
        self.n_dec_layers = gpt2_config.n_layer

        if pretrained_dir is not None:
            # gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_dir, config=gpt2_config)
            model = GPT2LMHeadModel(gpt2_config)
            state_dict = torch.load(pretrained_dir + 'pytorch_model.bin', map_location='cpu')
            gpt2 = load_weight(model, state_dict, freeze_pretrained=freeze_pretrained)
        else:
            gpt2 = GPT2LMHeadModel(gpt2_config)
        self.decoder = gpt2
        
        # self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, memory_embed, memory_mask, target_input_ids, target_input_mask, pos_embed=None):
        """ Calculate the decoder outputs given encoder memory, target (input_ids), and target mask.
          Args:
            memory_embed (B, N+L+1, d): the embeddings obtained by video encoder.
            target_mask (B, S): attention masks with flot values.
        """
        bsz, n_seq, d_emb = memory_embed.shape

        if pos_embed is not None:
            memory_embed = memory_embed + pos_embed
        
        logits, hs = self.decoder(
            input_ids = target_input_ids,
            attention_mask = target_input_mask,
            encoder_hidden_states = memory_embed,
            encoder_attention_mask = memory_mask,
            # labels = target_input_ids,
            return_dict = False
        )
        loss = self.loss(logits[:, :-1, :], target_input_ids[:, 1:], target_input_mask[:, 1:])

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
        caption_logits = torch.zeros(size=(bsz, self.max_dec_length, self.tokenizer.vocab_size), device=memory_embed.device)

        bsz_ends = torch.zeros(size=(bsz,), dtype=torch.int32)
        i = 0
        while not all(bsz_ends) and i < self.max_dec_length-1:
            logits, _ = self.forward(memory_embed, memory_mask, caption_ids, caption_mask, pos_embed)
            logits = logits[:, i, :]
            pred_ids = torch.argmax(logits, axis=-1) # (B, )

            caption_ids[:, i+1] = pred_ids
            caption_mask[:, i+1] = 1.0
            caption_logits[:, i, :] = logits

            _ends = (pred_ids == self.tokenizer.eos_token_id).detach().cpu().int()
            bsz_ends = torch.where(bsz_ends!=0, bsz_ends, _ends*(i+1))

            i += 1
        
        target_captions = list(map(lambda x: self.tokenizer.decode(x[1:-1], skip_special_tokens=True), caption_ids))

        loss = None
        if target_input_ids is not None and target_input_mask is not None:
            loss = self.loss(caption_logits[:, :-1, :], target_input_ids[:, 1:], target_input_mask[:, 1:])

        return (caption_ids, caption_logits, bsz_ends, target_captions, loss)
    
    
    def loss(self, logits, labels, mask=None) -> torch.tensor:
        """ Computes Causal Language Modeling (CLM) loss value according to the loss function.
          Args:
            logits:
            labels: 
        """
        bsz, seq_len, v = logits.shape
        # loss = self.loss_func(logits.view(-1, v), labels.view(-1))
        # loss = F.cross_entropy(logits.view(-1, v), labels.view(-1), reduce=False)
        loss = F.cross_entropy(logits.reshape(-1, v), labels.reshape(-1), reduce=False)
        # loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
        loss = torch.sum(loss * mask.reshape(-1)) / torch.sum(mask)
        return loss
    
    
    @classmethod
    def prepare_input(self, tokenizer, max_dec_length, batch_tokens=None, batch_size=None) -> Tuple:
        """ Prepare the inputs to decoder. Generate padded targets if the batch_tokens is not None, else generate the start tokens for infenrence.
        """
        assert batch_tokens or batch_size
        
        if batch_tokens is not None:
            # add bos and eos
            for i in range(len(batch_tokens)):
                batch_tokens[i] = "{}{}{}".format(tokenizer.bos_token, batch_tokens[i], tokenizer.eos_token)

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
            start_token = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            end_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            target_caption_ids = torch.zeros((batch_size, max_dec_length), dtype=torch.long)
            # target_caption_mask = torch.ones((batch_size, max_dec_length), dtype=torch.bool)
            target_caption_mask = torch.zeros((batch_size, max_dec_length), dtype=torch.long)

            target_caption_ids[:, 0] = start_token
            target_caption_mask[:, 0] = 1.0
            
        return target_caption_ids, target_caption_mask
            
    
    @classmethod
    def process_output(self, pred_ids, pred_lens, tokenizer):
        """ Get the output caption sentences given batch prediction indices.
        """
        result = []
        for i in range(len(pred_ids)):
            dec_sent = tokenizer.decode(pred_ids[i][:pred_lens[i]], skip_special_tokens=True)
            result.append(dec_sent)
        return result