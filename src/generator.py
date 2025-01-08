import torch
from torch.functional import F

from src.data import read_file
from src.vocab import TOKEN_EOS, TOKEN_SOS


class WordGenerator:
    def __init__(
            self,
            model,
            vocab,
            block_file=None,
            device="cpu",
            top_k=5,
            top_p=0.9,
            temperature=1.0
        ):
        self.model = model
        self.vocab = vocab
        self.block_words = self._init_block_words_from_file(block_file)
        self.device = device
        self.top_k = 0 if not top_k else top_k
        self.top_p = top_p
        self.temperature = temperature

    def _init_block_words_from_file(self, path):
        if not path:
            return []
        with open(path) as file:
            text = file.read().lower()
        names = text.splitlines()
        return names

    def _adjust_with_temperature(self, logits):
        temp = max(self.temperature, 1e-5)
        return logits / temp

    def _reduce_to_top_k(self, probs):
        k = self.top_k if self.top_k > 0 else probs.shape[0]
        topk_probs, topk_indices = torch.topk(probs, k=k)
        return topk_probs, topk_indices

    def _nucleus_sampling(self, sorted_probs, sorted_indices):
        if self.top_p >= 1.0:
            return sorted_probs, sorted_indices

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.top_p
        top_p_ix = top_p_mask.sum().item()

        filtered_probs = sorted_probs[:top_p_ix + 1]
        filtered_indices = sorted_indices[:top_p_ix + 1]

        return filtered_probs, filtered_indices

    def _generate_next_char(self, prefix, state):
        if not prefix:
            prefix = [TOKEN_SOS]

        # Convert prefix to indices
        input_indices = [self.vocab.char_to_ix[char] for char in prefix]
        input_tensor = torch.tensor(
            input_indices, dtype=torch.long).unsqueeze(0).to(self.device)

        # Forward pass
        logits, state = self.model(input_tensor, [len(input_indices)], state)

        # Logits form last time step's output
        logits = logits[:, -1, :]  # Shape: [1, vocab_size]
        logits = self._adjust_with_temperature(logits)

        probs = F.softmax(logits, dim=-1).squeeze(0)
        probs, indices = self._reduce_to_top_k(probs)
        probs, indices = self._nucleus_sampling(probs, indices)

        # Sample next character
        sampled_index = torch.multinomial(probs, num_samples=1).item()
        next_char_index = indices[sampled_index].item()
        next_char = self.vocab.ix_to_char[next_char_index]

        return next_char, state

    def _generate_word(self, prefix, min_length, max_length):
        state = self.model.init_state(batch_size=1)
        generated = "" if not prefix else prefix

        self.model.eval()

        with torch.no_grad():
            for _ in range(max_length):
                next_char, state = self._generate_next_char(generated, state)
                if next_char == TOKEN_EOS:
                    if len(generated) >= min_length:
                        break
                    else:
                        continue
                generated += next_char

        return generated

    def generate_word(self, prefix="", min_length=10, max_length=30):
        generated = self._generate_word(prefix, min_length, max_length)
        if generated not in self.block_words:
            return generated.capitalize()
        return self.generate_word(prefix, min_length, max_length)

    def validate_prefix(self, prefix):
        valid_chars = self.vocab.chars
        invalid_chars = []
        for char in prefix:
            if char not in valid_chars:
                invalid_chars.append(char)
        return invalid_chars
