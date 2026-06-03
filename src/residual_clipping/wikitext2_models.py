"""Language models for WikiText-2 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        *,
        tie_weights: bool,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if tie_weights:
            if hidden_size != embedding_size:
                raise ValueError("Weight tying requires hidden_size to equal embedding_size.")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(embeddings, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters())
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=weight.dtype),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=weight.dtype),
        )


def make_lstm_language_model(
    *,
    vocab_size: int,
    embedding_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    tie_weights: bool,
) -> LSTMLanguageModel:
    return LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=tie_weights,
    )
