from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from simple_parsing import Serializable
from tqdm import tqdm


@dataclass
class PhonemeTokenizer(Serializable):
    """
    A simple tokenizer that splits on a delimiter and adds a BOS token.
    The first item in the vocabulary is the BOS token and the second item is the UNK token.

    Attributes:
        phoneme_to_id (Dict[str, int]): Mapping from phoneme strings to unique IDs.
        delimiter (str): Character used to split phoneme strings.
    """

    # data fields
    phoneme_to_id: Dict[str, int] = field(default_factory=dict)
    delimiter: str = field(default=" ")

    # constant fields
    BOS_id: int = field(default=0, init=False)  # always 0
    UNK_id: int = field(default=1, init=False)  # always 1

    def __post_init__(self):
        # checks
        num_unique_ids = len(set(self.phoneme_to_id.values()))
        num_unique_tokens = len(self.phoneme_to_id)
        msg = "Phoneme to ID mapping must be one-to-one"
        assert num_unique_ids == num_unique_tokens, msg
        msg = "Delimiter must not be in vocabulary"
        assert self.delimiter not in self.phoneme_to_id, msg
        # cache the reverse mapping
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.phoneme_to_id)

    @property
    def BOS_token(self) -> str:
        return self.id_to_phoneme[self.BOS_id]

    @property
    def UNK_token(self) -> str:
        return self.id_to_phoneme[self.UNK_id]

    def encode(self, phoneme_string: str) -> torch.Tensor:
        """
        Encode a phoneme string into a tensor of IDs.
        The BOS token is always added to the beginning of the string.
        The UNK token is used for any phoneme that is not in the vocabulary.
        """
        phonemes = phoneme_string.split(self.delimiter)
        phonemes = [self.BOS_token] + phonemes
        ids = [self.phoneme_to_id.get(phoneme, self.UNK_id) for phoneme in phonemes]
        return torch.tensor(ids)

    def decode(self, ids: torch.Tensor) -> str:
        """
        Decode a tensor of IDs into a phoneme string.
        The BOS token is removed from the ids.
        The UNK token is used for any ID that is not in the vocabulary.
        """
        ids = ids[ids != self.BOS_id].cpu().tolist()
        phonemes = [self.id_to_phoneme.get(id, self.UNK_token) for id in ids]
        phoneme_string = self.delimiter.join(phonemes)
        return phoneme_string

    @classmethod
    def train_from_strings_dir(
        cls,
        strings_dir: str,
        pattern: str = "**/*.txt",
        delimiter: str = " ",
        bos_token: str = "BOS",
        unk_token: str = "UNK",
    ) -> "PhonemeTokenizer":
        """
        Train a PhonemeTokenizer from text files in a specified directory.

        This method reads all text files matching the given pattern in the specified
        directory, extracting phonemes from each file. Each text file should contain
        a single line of text. The method constructs a vocabulary of phonemes,
        assigning a unique ID to each phoneme, with the first ID reserved for the
        beginning-of-sequence (BOS) token and the second for the unknown (UNK) token.

        Args:
            strings_dir (str): The directory containing the text files.
            pattern (str): The glob pattern to match text files (default is "**/*.txt").
            delimiter (str): The delimiter used to split phonemes in the text (default is " ").
            bos_token (str): The token to use for the beginning of the sequence (default is "BOS").
            unk_token (str): The token to use for unknown phonemes (default is "UNK").

        Returns:
            PhonemeTokenizer: An instance of PhonemeTokenizer trained on the provided text files.
        """
        strings_paths = list(Path(strings_dir).glob(pattern))
        phonemes = set()
        for string_path in tqdm(strings_paths, desc="Training tokenizer"):
            phonemes.update(string_path.read_text().split(delimiter))
        phonemes_sorted = [bos_token, unk_token] + sorted(list(phonemes))
        phoneme_to_id = {phoneme: i for i, phoneme in enumerate(phonemes_sorted)}
        return cls(
            phoneme_to_id=phoneme_to_id,
            delimiter=delimiter,
        )

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "PhonemeTokenizer":
        """
        Load a PhonemeTokenizer from a specified model directory.

        This method retrieves the tokenizer configuration from a json file located
        in the specified model directory. It expects the json file to be named
        "tokenizer.json".

        Args:
            model_dir (str): The directory containing the model files, including the tokenizer configuration.

        Returns:
            PhonemeTokenizer: An instance of PhonemeTokenizer loaded from the specified model directory.
        """

        model_dir = Path(model_dir)
        tokenizer_path: Path = model_dir / "tokenizer.json"
        msg = f"Tokenizer file {tokenizer_path} does not exist"
        assert tokenizer_path.exists(), msg
        return cls.load_json(tokenizer_path)
