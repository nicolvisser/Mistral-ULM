from typing import Tuple

import torch

from ulm import TransformerModel, TransformerModelArgs, __version__
from ulm.units import UnitTokenizer
from ulm.phones import PhonemeTokenizer
import requests

repository_url = "https://github.com/nicolvisser/Mistral-ULM/"
release_url = repository_url + "releases/download/v" + __version__ + "/"

# fmt: off
ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k_urls = {
    (100, 0): release_url + "mistral-ulm-k-100-lmbda-0-hours-1k-steps-10k-b7a35cc5.pt",
    (100, 600): release_url + "mistral-ulm-k-100-lmbda-600-hours-1k-steps-10k-17e91718.pt",
    (100, 1500): release_url + "mistral-ulm-k-100-lmbda-1500-hours-1k-steps-10k-6ba3cc55.pt",
    (100, 3000): release_url + "mistral-ulm-k-100-lmbda-3000-hours-1k-steps-10k-5d0669b5.pt",
    (100, 5000): release_url + "mistral-ulm-k-100-lmbda-5000-hours-1k-steps-10k-0c1af6fe.pt",
    (100, 9000): release_url + "mistral-ulm-k-100-lmbda-9000-hours-1k-steps-10k-92bc9c97.pt",
    (200, 0): release_url + "mistral-ulm-k-200-lmbda-0-hours-1k-steps-10k-5574c355.pt",
    (200, 700): release_url + "mistral-ulm-k-200-lmbda-700-hours-1k-steps-10k-487bae0c.pt",
    (200, 1500): release_url + "mistral-ulm-k-200-lmbda-1500-hours-1k-steps-10k-8524865e.pt",
    (200, 3000): release_url + "mistral-ulm-k-200-lmbda-3000-hours-1k-steps-10k-ac40635d.pt",
    (200, 5000): release_url + "mistral-ulm-k-200-lmbda-5000-hours-1k-steps-10k-09a1f87c.pt",
    (200, 7500): release_url + "mistral-ulm-k-200-lmbda-7500-hours-1k-steps-10k-b685aac0.pt",
    (500, 0): release_url + "mistral-ulm-k-500-lmbda-0-hours-1k-steps-10k-532387cd.pt",
    (500, 600): release_url + "mistral-ulm-k-500-lmbda-600-hours-1k-steps-10k-b96bee1d.pt",
    (500, 1500): release_url + "mistral-ulm-k-500-lmbda-1500-hours-1k-steps-10k-853fbee4.pt",
    (500, 2800): release_url + "mistral-ulm-k-500-lmbda-2800-hours-1k-steps-10k-ef015b7f.pt",
    (500, 4500): release_url + "mistral-ulm-k-500-lmbda-4500-hours-1k-steps-10k-9b75c46e.pt",
    (500, 7000): release_url + "mistral-ulm-k-500-lmbda-7000-hours-1k-steps-10k-886ec1dd.pt",
    (1000, 0): release_url + "mistral-ulm-k-1000-lmbda-0-hours-1k-steps-10k-56124eaa.pt",
    (1000, 600): release_url + "mistral-ulm-k-1000-lmbda-600-hours-1k-steps-10k-810cb2f0.pt",
    (1000, 1400): release_url + "mistral-ulm-k-1000-lmbda-1400-hours-1k-steps-10k-7f014b7e.pt",
    (1000, 2500): release_url + "mistral-ulm-k-1000-lmbda-2500-hours-1k-steps-10k-c5cd03ce.pt",
    (1000, 3800): release_url + "mistral-ulm-k-1000-lmbda-3800-hours-1k-steps-10k-2c798110.pt",
    (1000, 6000): release_url + "mistral-ulm-k-1000-lmbda-6000-hours-1k-steps-10k-df9e72a7.pt",
}

ulm_wavlm_layer_11_dpdp_hours_60k_steps_200k_urls = {
    (500, 4500): release_url + "mistral-ulm-k-500-lmbda-4500-hours-60k-steps-200k-09759305.pt",
}
# fmt: on


def _lm_from_url(
    checkpoint_url: str, map_location="cpu", progress=True
) -> TransformerModel:
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url,
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    model_args = TransformerModelArgs.from_dict(checkpoint["model_args"])
    model = TransformerModel(model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ULM model with {num_params:,} parameters.")
    return model, tokenizer


def ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k(
    k: int, lmbda: int, map_location="cpu", progress=True
) -> Tuple[TransformerModel, UnitTokenizer]:
    msg = f"Pretrained ULM model for (k, lmbda) = ({k}, {lmbda}) not found. Available models: {ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k_urls.keys()}"
    assert (k, lmbda) in ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k_urls, msg
    checkpoint_url = ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k_urls[(k, lmbda)]
    model = _lm_from_url(checkpoint_url, map_location, progress)
    tokenizer = UnitTokenizer()
    return model, tokenizer


def ulm_wavlm_layer_11_dpdp_hours_60k_steps_200k(
    k: int, lmbda: int, map_location="cpu", progress=True
) -> Tuple[TransformerModel, UnitTokenizer]:
    msg = f"Pretrained ULM model for (k, lmbda) = ({k}, {lmbda}) not found. Available models: {ulm_wavlm_layer_11_dpdp_hours_60k_steps_200k_urls.keys()}"
    assert (k, lmbda) in ulm_wavlm_layer_11_dpdp_hours_60k_steps_200k_urls, msg
    checkpoint_url = ulm_wavlm_layer_11_dpdp_hours_60k_steps_200k_urls[(k, lmbda)]
    model = _lm_from_url(checkpoint_url, map_location, progress)
    tokenizer = UnitTokenizer()
    return model, tokenizer


def plm_us_arpa_hours_1k_steps_10k(
    k: int, lmbda: int, map_location="cpu", progress=True
) -> Tuple[TransformerModel, UnitTokenizer]:
    model = _lm_from_url(
        "https://github.com/nicolvisser/Mistral-ULM/releases/download/v0.1.0/mistral-plm-hours-1k-steps-10k-6d3a38a9.pt",
        map_location,
        progress,
    )
    tokenizer_url = "https://github.com/nicolvisser/Mistral-ULM/releases/download/v0.1.0/us_arpa_tokenizer.json"
    try:
        tokenizer_response = requests.get(tokenizer_url)
        tokenizer_response.raise_for_status()
        tokenizer_data = tokenizer_response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download tokenizer from {tokenizer_url}: {e}")
    except ValueError as e:
        raise RuntimeError(f"Failed to parse tokenizer JSON data: {e}")
    tokenizer = PhonemeTokenizer.from_dict(tokenizer_data)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = plm_us_arpa_hours_1k_steps_10k(100, 0)
    print(tokenizer.encode("H EY1"))
