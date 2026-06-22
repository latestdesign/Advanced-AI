import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """
    BLIP-2 style modality projector (simplified Q-Former).

    Input:
        x : [B, 1024, 768]   (ViT patch embeddings)

    Output:
        [B, num_queries, 960]
    """

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.vit.hidden_dim      # 768
        self.lm_hidden_dim = cfg.lm.hidden_dim   # 960

   
        self.num_queries = cfg.projector.image_token_length

        self.qformer_dim = self.hidden_dim

        # Learned query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_queries, self.qformer_dim)
        )

        # Cross-attention: queries attend to ViT features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.qformer_dim,
            num_heads=8,
            batch_first=True,
            kdim=self.hidden_dim,
            vdim=self.hidden_dim,
        )

        # Self-attention among query tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.qformer_dim,
            num_heads=8,
            batch_first=True,
        )

        # Feed-forward block
        self.mlp = nn.Sequential(
            nn.Linear(self.qformer_dim, 4 * self.qformer_dim),
            nn.GELU(),
            nn.Linear(4 * self.qformer_dim, self.qformer_dim),
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(self.qformer_dim)
        self.norm2 = nn.LayerNorm(self.qformer_dim)
        self.norm3 = nn.LayerNorm(self.qformer_dim)

        # Final projection to LM dimension
        self.proj_to_lm = nn.Linear(
            self.qformer_dim,
            self.lm_hidden_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 1024, 768]

        Returns:
            [B, num_queries, 960]
        """

        B = x.size(0)

        # [B, num_queries, 768]
        queries = self.query_tokens.expand(B, -1, -1)

        
        cross_out, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            need_weights=False,
        )

        queries = self.norm1(queries + cross_out)

        self_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
            need_weights=False,
        )

        queries = self.norm2(queries + self_out)

        # Feed-forward
        mlp_out = self.mlp(queries)

        queries = self.norm3(queries + mlp_out)
        queries = self.proj_to_lm(queries)

        return queries