import math
import torch
from typing import Tuple, Dict
from torch import nn
import torch.nn.functional as F
from torch.nn import Transformer
import numpy as np
from buildings_bench.models.base_model import BaseModel


class TokenEmbedding(nn.Module):
    """Helper Module to convert tensor of input
       indices into corresponding tensor of token embeddings.
    """
    def __init__(self, vocab_size: int, emb_size: int):
        """
        Args:
            vocab_size (int): number of quantized load values in the entire vocabulary.
            emb_size (int): embedding size.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to the token embedding to
       introduce a notion of order within a time-series.
    """
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 500):
        """
        Args:
            emb_size (int): embedding size.
            dropout (float): dropout rate.
            maxlen (int): maximum possible length of the incoming time series.
        """
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # batch first - use size(1)
        # need to permute token embeddings from [batch_size, seqlen x emb_size] to [seqlen x batch_size, emb_size]
        return self.dropout(token_embedding.permute(1,0,2) + self.pos_embedding[:token_embedding.size(1), :]).permute(1,0,2)


class TimeSeriesSinusoidalPeriodicEmbedding(nn.Module):
    """This module produces a sinusoidal periodic embedding for a sequence of values in [-1, +1]."""
    def __init__(self, embedding_dim: int) -> None:
        """
        Args:
            embedding_dim (int): embedding size.
        """
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is expected to be [batch_size, seqlen, 1]."""
        with torch.no_grad():
            x = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=2)
        # [batch_size, seqlen x 2] --> [batch_size, seqlen, embedding_dim]
        return self.linear(x)


class ZeroEmbedding(nn.Module):
    """ Outputs zeros of the desired output dim."""
    def __init__(self, embedding_dim: int): 
        """
        Args:
            embedding_dim (int): embedding size.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.zeros_embedding = nn.Parameter(
            torch.zeros(1, 1, embedding_dim), requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ `x` is expected to be [batch_size, seqlen, 1]."""
        return self.zeros_embedding.repeat(x.shape[0], x.shape[1], 1)
    

class LoadForecastingTransformer(BaseModel):
    """
    An encoder-decoder time series Transformer. Based on PyTorch nn.Transformer.

    - Uses masking in the decoder to prevent the model from peeking into the future
    - Uses N(0, 0.02) for weight initialization
    - Trains with teacher forcing (i.e. the target is used as the input to the decoder)
    - continuous_loads (True) just predict target values
                     (False) categorical over quantized load values
    """
    def __init__(self,
                 context_len: int = 168,
                 pred_len: int = 24,
                 vocab_size = 2274,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 256,
                 dropout: float = 0.0,
                 activation: str = 'gelu',
                 continuous_loads = False,
                 continuous_head = 'mse',
                 ignore_spatial = False):
        """
        Args:
            context_len (int): length of the input sequence.
            pred_len (int): length of the output sequence.
            vocab_size (int): number of quantized load values in the entire vocabulary.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            d_model (int): number of expected features in the encoder/decoder inputs.
            nhead (int): number of heads in the multi-head attention models.
            dim_feedforward (int): dimension of the feedforward network model.
            dropout (float): dropout value.
            activation (str): the activation function of encoder/decoder intermediate layer, relu or gelu.
            continuous_loads (bool): whether inputs are continuous/to train the model to predict continuous values.
            continuous_head (str): 'mse' or 'gaussian_nll'.
            ignore_spatial (bool): whether to ignore the spatial features.
        """
        super().__init__(context_len, pred_len, continuous_loads)
        
        self.continuous_head = continuous_head
        self.vocab_size = vocab_size
        self.ignore_spatial = ignore_spatial
        s = d_model // 256

        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       activation=activation,
                                       batch_first=True)

        if self.continuous_loads:
            out_dim = 1 if self.continuous_head == 'mse' else 2
            self.logits = nn.Linear(d_model, out_dim)
            self.power_embedding = nn.Linear(1, 64 * s)
        else:
            self.logits = nn.Linear(d_model, self.vocab_size)
            self.power_embedding = TokenEmbedding(self.vocab_size, 64 * s)
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.pred_len)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)
        self.building_embedding = nn.Embedding(2, 32 * s)
        self.lat_embedding = nn.Linear(1, 32 * s)
        self.lon_embedding = nn.Linear(1, 32 * s)
        if self.ignore_spatial:
            self.lat_embedding = ZeroEmbedding(32 * s)
            self.lon_embedding = ZeroEmbedding(32 * s)
        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s)
        
    def to(self, device):
        self.tgt_mask = self.tgt_mask.to(device)
        return super().to(device)

    def forward(self, x):
        # [batch_size, seq_len, d_model]
        time_series_embed = torch.cat([
            self.lat_embedding(x['latitude']),
            self.lon_embedding(x['longitude']),
            self.building_embedding(x['building_type']).squeeze(2),
            self.day_of_year_encoding(x['day_of_year']),
            self.day_of_week_encoding(x['day_of_week']),
            self.hour_of_day_encoding(x['hour_of_day']),
            self.power_embedding(x['load']).squeeze(2),
        ], dim=2)
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        # [batch_size, pred_len, d_model]
        # The last element of the target sequence is not used as input
        # The last element of the source sequence is used as the initial decoder input
        tgt_series_inputs = time_series_embed[:, self.context_len-1 : -1, :]
        src_series_embed = self.positional_encoding(src_series_inputs)
        tgt_series_embed = self.positional_encoding(tgt_series_inputs) 

        # The output of TransformerEncoder is the sequence from the last layer
        # The shape will be [batch_size, context_len, d_model]
        
        memory = self.transformer.encoder(src_series_embed, mask=None)
        outs = self.transformer.decoder(tgt_series_embed, memory, tgt_mask=self.tgt_mask)
        return self.logits(outs)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generate_sample(x, greedy=True)
    
    def loss(self, x, y):
        if self.continuous_loads and self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_loads and self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)
        else:
            return F.cross_entropy(x.reshape(-1, self.vocab_size),
                                             y.long().reshape(-1))
                    
    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.logits.requires_grad_(True)
        return self.logits.parameters()
        #return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        stored_ckpt = torch.load(checkpoint_path)
        model_state_dict = stored_ckpt['model']
        new_state_dict = {}
        for k,v in model_state_dict.items():
            # remove string 'module.' from the key
            if 'module.' in k:
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)    
        print(f"Loaded model checkpoint from {checkpoint_path}...")
    

    @torch.no_grad()
    def generate_sample(self, 
                 x,
                 temperature=1.0,
                 greedy=False,
                 num_samples=1,
                 top_k=0,
                 top_p=1.0):
        """Sample from the conditional distribution.

        Use output of decoder at each prediction step as input to the next decoder step.
        Implements greedy decoding and random temperature-controlled sampling.
        
        Top-k sampling and nucleus sampling are deprecated.

        Args:
            x (Dict): dictionary of input tensors
            temperature (float): temperature for sampling
            greedy (bool): whether to use greedy decoding
            num_samples (int): number of samples to generate
            top_k (int): top k sampling (DEPRECATED)
            top_p (float): nucleus sampling (DEPRECATED)
        
        Returns:
            predictions (torch.Tensor): of shape [batch_size, pred_len, 1] or shape [batch_size, num_samples, pred_len] if num_samples > 1.
            distribution_parameters (torch.Tensor): of shape [batch_size, pred_len, 1]. Not returned if sampling.
        """
        time_series_embed = torch.cat([
            self.lat_embedding(x['latitude']),
            self.lon_embedding(x['longitude']),
            self.building_embedding(x['building_type']).squeeze(2),
            self.day_of_year_encoding(x['day_of_year']),
            self.day_of_week_encoding(x['day_of_week']),
            self.hour_of_day_encoding(x['hour_of_day']),
            self.power_embedding(x['load']).squeeze(2),
        ], dim=2)
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        tgt_series_inputs = time_series_embed[:, self.context_len-1 : -1, :]
        src_series_embed = self.positional_encoding(src_series_inputs)

        encoder_output = self.transformer.encoder(src_series_embed)
        decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)
        if num_samples > 1 and not greedy:
            # [batch_size, 1, emb_size] --> [batch_size * num_sampes, 1, emb_size]
            decoder_input = decoder_input.repeat_interleave(num_samples, dim=0)
            encoder_output = encoder_output.repeat_interleave(num_samples, dim=0)
        all_preds, all_logits = [], []
        for k in range(1, self.pred_len+1):
            decoder_embed = self.positional_encoding(decoder_input)
            tgt_mask = self.transformer.generate_square_subsequent_mask(k)
            decoder_output = self.transformer.decoder(decoder_embed, encoder_output, tgt_mask.to(encoder_output.device))
            # [batch_size, 1] if continuous (2 if head is gaussian_nll) or [batch_size, vocab_size] if not continuous_loads
            outputs = self.logits(decoder_output[:, -1, :])
            all_logits += [outputs.unsqueeze(1)]

            if self.continuous_loads:
                if self.continuous_head == 'mse':
                    all_preds += [outputs] 
                elif self.continuous_head == 'gaussian_nll':
                    if greedy:
                        all_preds += [outputs[:, 0].unsqueeze(1)] # mean only
                        outputs = all_preds[-1] # [batch_size, 1, 1]
                    else:
                        mean = outputs[:,0]
                        std= torch.nn.functional.softplus(outputs[:,1])
                        outputs = torch.distributions.normal.Normal(mean, std).sample().unsqueeze(1)
                        all_preds += [outputs]    

            elif not greedy:
                # outputs are [batch_size, vocab_size]
                if top_k > 0 or top_p < 1.0:
                    raise NotImplementedError("Top-k and nucleus sampling are not supported")
                    # Apply top-k and/or nucleus filtering
                    outputs = beam_search.top_k_top_p_filtering(
                        outputs, top_k=top_k, top_p=top_p
                    )
                # Sample from a Categorical distribution with logits outputs
                all_preds += [torch.multinomial(torch.nn.functional.softmax(outputs/temperature, dim=1), 1)]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1] # [batch_size * num_samples, 1]
            else:
                # outputs are [batch_size, vocab_size]
                # Greedy decoding
                all_preds += [outputs.argmax(dim=1).unsqueeze(1)]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1]
                
            # [batch_size, d_model]
            if k < self.pred_len:
                # [batch_size, d_model]
                next_decoder_input = tgt_series_inputs[:, k]
                if num_samples > 1 and not greedy:
                    # [batch_size, d_model] --> [batch_size * num_samples, d_model]
                    next_decoder_input = next_decoder_input.repeat_interleave(num_samples, dim=0)
                # Use the embedding predicted load instead of the ground truth load
                embedded_pred = self.power_embedding(outputs)  
                if not self.continuous_loads:
                    # [batch_size, 1, 1, 64*s] --> [batch_size, 64*s]
                    embedded_pred = embedded_pred.squeeze(2).squeeze(1)
                next_decoder_input = torch.cat([ next_decoder_input[:, :-embedded_pred.shape[-1]], embedded_pred ], dim=1)
                # Append the next decoder input to the decoder input
                decoder_input = torch.cat([decoder_input, next_decoder_input.unsqueeze(1)], dim=1)
        if num_samples == 1 or greedy:
            if self.continuous_head == 'gaussian_nll':
                # [batch_size, pred_len, 2]
                gaussian_params = torch.stack(all_logits,1)[:,:,0,:]
                means = gaussian_params[:,:,0]
                sigma = torch.nn.functional.softplus(gaussian_params[:,:,1])
                return torch.stack(all_preds,1), torch.cat([means.unsqueeze(2), sigma.unsqueeze(2)],2)
            else:
                return torch.stack(all_preds,1), torch.stack(all_logits,1)[:,:,0,:]
        else:
            # [batch_size, num_samples, pred_len]
            return torch.stack(all_preds,1).reshape(-1, num_samples, self.pred_len)


    # @torch.no_grad()
    # def beam_search(
    #     self,
    #     x: torch.Tensor,
    #     num_beams: int,
    #     num_return_sequences: int
    # ):
    #     r"""
    #     Generates sequences for models with a language modeling head using beam search decoding.

    #     Parameters:

    #         x (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
    #             The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
    #             :obj:`torch.LongTensor` of shape :obj:`(1,)`.
    #         beam_scorer (:obj:`BeamScorer`):
    #             An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
    #             constructed, stored and sorted during generation. For more information, the documentation of
    #             :class:`~transformers.BeamScorer` should be read.

    #     Return:
    #         :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
    #         sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
    #         batches finished early due to the :obj:`eos_token_id`.
    #     """
    #     beam_scorer = beam_search.BeamSearchScorer(
    #         batch_size=x['latitude'].shape[0],
    #         max_length=self.pred_len,
    #         num_beams=num_beams,
    #         device=x['latitude'].device,
    #         length_penalty=1.0,
    #         do_early_stopping=False,
    #         num_beam_hyps_to_keep=num_return_sequences,
    #     )

    #     time_series_embed = torch.cat([
    #         self.lat_embedding(x['latitude']),
    #         self.lon_embedding(x['longitude']),
    #         self.building_embedding(x['building_type']).squeeze(2),
    #         self.day_of_year_encoding(x['day_of_year']),
    #         self.day_of_week_encoding(x['day_of_week']),
    #         self.hour_of_day_encoding(x['hour_of_day']),
    #         self.power_embedding(x['load']).squeeze(2),
    #     ], dim=2)
    #     # [batch_size, context_len, emb_size]
    #     src_series_inputs = time_series_embed[:, :self.context_len, :]
    #     tgt_series_inputs = time_series_embed[:, self.context_len-1 : -1, :]
    #     src_series_embed = self.positional_encoding(src_series_inputs)

    #     encoder_output = self.transformer.encoder(src_series_embed)
    #     decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)

    #     batch_size = len(beam_scorer._beam_hyps)
    #     num_beams = beam_scorer.num_beams

    #     if num_beams > 1:
    #         # [batch_size, 1, d_model] --> [batch_size * num_beams, 1, d_model]
    #         decoder_input = decoder_input.repeat_interleave(num_beams, dim=0)
    #         encoder_output = encoder_output.repeat_interleave(num_beams, dim=0)

    #     batch_beam_size, cur_len, _ = decoder_input.shape

    #     assert cur_len == 1, \
    #            f"expected `decoder_input` to have 1 tokens, but has {cur_len}"
    #     assert (
    #         num_beams * batch_size == batch_beam_size
    #     ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    #     beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=decoder_input.device)
    #     beam_scores[:, 1:] = -1e9
    #     beam_scores = beam_scores.view((batch_size * num_beams,))
    #     decoded_tokens = []
    #     while cur_len < self.pred_len+1:
    #         decoder_embed = self.positional_encoding(decoder_input)
    #         tgt_mask = self.transformer.generate_square_subsequent_mask(cur_len)
    #         decoder_output = self.transformer.decoder(decoder_embed, encoder_output, tgt_mask.to(encoder_output.device))
    #         # [batch_size * num_beams, vocab_size] 
    #         next_token_logits = self.logits(decoder_output[:, -1, :])
    #         next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

    #         next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    #         # reshape for beam search
    #         vocab_size = next_token_scores.shape[-1]
    #         next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    #         next_token_scores, next_tokens = torch.topk(
    #             next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    #         )

    #         next_indices = next_tokens // vocab_size
    #         next_tokens = next_tokens % vocab_size

    #         # stateless
    #         beam_outputs = beam_scorer.process(
    #             decoder_input,
    #             next_token_scores,
    #             next_tokens,
    #             next_indices
    #         )
    #         beam_scores = beam_outputs["next_beam_scores"]
    #         beam_next_tokens = beam_outputs["next_beam_tokens"]
    #         beam_idx = beam_outputs["next_beam_indices"]
    #         decoded_tokens += [beam_next_tokens]
                       
    #         if cur_len < self.pred_len:
    #             # [batch_size * num_beams, cur_len, embd_sz]
    #             decoder_input = decoder_input[beam_idx, :]
    #             # [batch_size * num_beams, embd_sz]
    #             beam_next_token_embeddings = self.power_embedding(beam_next_tokens)                  
    #             # covariates for next step
    #             next_step_covariates = tgt_series_inputs[:, cur_len]
    #             if num_beams > 1:
    #                 next_step_covariates = next_step_covariates.repeat_interleave(num_beams, dim=0)
    #             next_decoder_input = torch.cat([ next_step_covariates[:, :-beam_next_token_embeddings.shape[-1]], beam_next_token_embeddings ], dim=1)
    #             # Append the next decoder input to the decoder input
    #             decoder_input = torch.cat([decoder_input, next_decoder_input.unsqueeze(1)], dim=1)
            
    #         cur_len = cur_len + 1

    #         # model_kwargs = self._update_model_kwargs_for_generation(
    #         #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         # )
    #         # if model_kwargs["past"] is not None:
    #         #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

    #         if beam_scorer.is_done:
    #             break
        
    #     # [batch_size * num_beams, pred_len]
    #     decoded_tokens = torch.stack(decoded_tokens,1)
    #     decoded_tokens = beam_scorer.finalize(
    #         decoded_tokens, beam_scores
    #     )

    #     # [batch_size, pred_len]
    #     return decoded_tokens
