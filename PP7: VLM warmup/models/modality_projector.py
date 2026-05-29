"""Exercise stub for the PP7 modality projector."""
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, vision_hidden_dim, language_hidden_dim, **kwargs):
        """Placeholder initializer for the exercise implementation.

        Args:
            *args: Positional arguments the student-defined projector may need.
            **kwargs: Keyword arguments the student-defined projector may need.
        """
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.language_hidden_dim = language_hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_dim, language_hidden_dim),
            nn.GELU(),
            nn.Linear(language_hidden_dim, language_hidden_dim),
        )

    def forward(self, image_embd):
        return self.projector(image_embd)
