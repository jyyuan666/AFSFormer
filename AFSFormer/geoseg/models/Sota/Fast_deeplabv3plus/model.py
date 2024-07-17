"""Module providing a class to define the Fire Segmentation Model."""
from typing import Tuple
import torch
from torch import nn

from geoseg.models.Sota.Fast_deeplabv3plus.modules import Encoder, Decoder


class FireSegmentationModel(nn.Module):
    """The Fire Segmentation Model.

    Attributes
    ----------
    encoder : Encoder
        The encoder module
    decoder : Decoder
        The encoder module
    """
    def __init__(self, input_size: Tuple[int, int], device: str) -> None:
        """Initialize the Fire Segmentation Model.

        Parameters
        ----------
        input_size : (int, int)
            The size of the input image.
        device : str
            The device where the model will be loaded.
        """
        super().__init__()
        # Set the encoder module.
        self.encoder = Encoder()
        # Set the decoder module.
        self.decoder = Decoder(target_size=input_size)
        # Assign the model to the desired device.
        self.to(device)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the Fire Segmentation Model.

        Parameters
        ----------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The output tensor containing the final segmentation mask of
            the background and the foreground.
        """
        # Apply the encoder block to the input and get the
        # intermediate features.
        f1, f2, f3, f4 = self.encoder(x)
        # Decode the final segmentation mask.
        out = self.decoder(f1, f2, f3, f4)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FireSegmentationModel(input_size=(1024, 1024), device=device)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))



