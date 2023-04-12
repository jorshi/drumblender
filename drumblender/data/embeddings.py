"""
Audio embedding models that are pre-computed on datasets.
"""
from typing import Callable
from typing import Literal
from typing import Union

import torch
from einops import rearrange

try:
    import openl3
except ImportError:
    openl3 = None


class OpenL3:
    """
    Wrapper for OpenL3

    Cramer, Jason, et al. "Look, listen, and learn more: Design choices for
    deep audio embeddings." ICASSP 2019-2019 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.

    Because the openl3 library is not a dependency of this package, and because we
    still want to be able to instantiate this class for configuration purposes, we
    just raise an ImportError if the library is not installed during the call.

    Args:
        sample_rate: Sample rate of the input audio.
        embedding_size: Size of the embedding to extract.
        content_type: Type of content to extract embeddings for.
        input_repr: Input representation to use
        hop_size: Hop size to use for the embedding.
        summarize: Function to use to summarize the embedding over temporal dimension.
    """

    def __init__(
        self,
        sample_rate: int,
        embedding_size: Literal[512, 6144] = 6144,
        content_type: Literal["env", "music"] = "music",
        input_repr: Literal["linear", "mel128", "mel256"] = "mel128",
        hop_size: float = 0.1,
        summarize: Union[Callable, Literal["flatten", "mean"]] = "mean",
    ):
        self.sample_rate = sample_rate
        self.embedding_size = embedding_size
        self.content_type = content_type
        self.input_repr = input_repr
        self.hop_size = hop_size
        self.summarize = summarize

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if openl3 is None:
            raise ImportError(
                "OpenL3 is not installed, please install is with `pip install openl3`"
            )
        assert (
            x.ndim == 2 and x.shape[0] == 1
        ), "Expected input to have shape (1, samples)"

        # OpenL3 expects the input to be (samples, channels), only expect mono here
        x = rearrange(x, "1 n -> n 1")
        device = x.device
        embedding, _ = openl3.get_audio_embedding(
            x.detach().cpu().numpy(),
            self.sample_rate,
            embedding_size=self.embedding_size,
            content_type=self.content_type,
            input_repr=self.input_repr,
            hop_size=self.hop_size,
            center=True,
            verbose=0,
        )
        x = torch.from_numpy(embedding).to(device)

        if self.summarize == "flatten":
            x = x.flatten()
        elif self.summarize == "mean":
            x = x.mean(dim=0)
        else:
            x = self.summarize(x)

        return x
