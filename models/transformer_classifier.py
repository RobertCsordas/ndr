import torch
import torch.nn
import torch.nn.functional as F
import framework
from layers import TiedEmbedding
from layers.transformer import TransformerEncoderWithLayer, AttentionMask, Transformer
from typing import Callable, Optional
import math
from .encoder_decoder import add_eos
from layers.transformer.multi_head_attention import MultiHeadAttention


class TransformerClassifierModel(torch.nn.Module):
    def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 max_len=5000, transformer=TransformerEncoderWithLayer(),
                 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
                 out_mode: str = "linear", embedding_init: str = "pytorch", scale_mode: str = "none", 
                 result_column: str = "first", sos: bool = True, eos: bool = True, 
                 autoregressive: bool = False, **kwargs):
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        assert (out_mode != "tied") or (n_input_tokens == n_out_tokens)
        assert out_mode in ["tied", "linear", "attention"]

        self.out_mode = out_mode

        self.encoder_eos = n_input_tokens if eos else None
        self.encoder_sos = (n_input_tokens + 1) if sos else None
        self.state_size = state_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.n_out_tokens = n_out_tokens
        self.scale_mode = scale_mode
        self.autoregressive = autoregressive
        self.pos = pos_embeddig or framework.layers.PositionalEncoding(state_size, max_len=max_len, batch_first=True,
                                        scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0)

        self.result_column = result_column
        assert self.result_column in ["first", "last"]
        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)
        self.reset_parameters()

    def pos_embed(self, t: torch.Tensor, offset: int, scale_offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def construct(self, transformer, **kwargs):
        self.embedding = torch.nn.Embedding(self.n_input_tokens + 2, self.state_size)

        if self.out_mode == "tied":
            self.output_map = TiedEmbedding(self.embedding.weight, batch_dim=0)
        elif self.out_mode == "linear":
            self.output_map = torch.nn.Linear(self.state_size, self.n_out_tokens)
        elif self.out_mode == "attention":
            self.output_map = MultiHeadAttention(self.state_size, 1, out_size=self.n_out_tokens)
            self.out_query = torch.nn.Parameter(torch.randn([1, self.state_size]) / math.sqrt(self.state_size))

        self.trafo = transformer(d_model=self.state_size, dim_feedforward=int(self.ff_multiplier * self.state_size),
                                 **kwargs)


    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.embedding.weight)

        if self.output_map == "linear":
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def get_result(self, res: torch.Tensor, src_len: torch.Tensor) -> torch.Tensor:
        if self.result_column == "first":
            return res[:, 0]
        elif self.result_column == "last":
            return res.gather(1, src_len.view([src_len.shape[0], 1, 1]).expand(-1, -1, res.shape[-1]) - 1).squeeze(1)
        else:
            assert False

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> torch.Tensor:
        '''
        Run transformer encoder-decoder on some input/output pair

        :param src: source tensor. Shape: [N, S], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :return: prediction of the target tensor. Shape [N, T, C_out]
        '''

        if self.encoder_eos is not None:
            src = add_eos(src, src_len, self.encoder_eos, batch_dim=0)
            src_len = src_len + 1

        if self.encoder_sos is not None:
            src = F.pad(src, (1, 0), value=self.encoder_sos)
            src_len = src_len + 1

        src = self.pos_embed(self.embedding(src.long()), 0, 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)

        causality_mask = Transformer.generate_square_subsequent_mask(src.shape[1], src.device)\
                         if self.autoregressive else None

        res = self.trafo(src, AttentionMask(in_len_mask, causality_mask))

        if self.out_mode == "attention":
            return self.output_map(self.out_query.expand(src.shape[0], -1, -1), res, mask=AttentionMask(in_len_mask, None)).squeeze(1)
        else:
            return self.output_map(self.get_result(res, src_len))
