import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    LSTM Decoder for generating captions.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size (int): The size of the embedding vector.
            hidden_size (int): The number of features in the hidden state.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of recurrent layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Forward pass for training with teacher forcing.

        Args:
            features (torch.Tensor): Image features from the encoder (batch_size, embed_size).
            captions (torch.Tensor): Ground truth captions (batch_size, seq_length).

        Returns:
            torch.Tensor: Predicted logits for each word in the vocabulary.
        """
        # Remove the <end> token from captions for input
        captions = captions[:, :-1]
        embeddings = self.embed(captions)

        # Prepend image features to the caption embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)

        # Pass through the linear layer
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Generate captions for inference (greedy decoding).

        Args:
            inputs (torch.Tensor): Image features prepared for the LSTM (batch_size, 1, embed_size).
            states (tuple, optional): Initial hidden and cell states. Defaults to None.
            max_len (int, optional): Maximum length of the generated caption. Defaults to 20.

        Returns:
            list: A list of word indices representing the generated caption.
        """
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())

            # Prepare the next input
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

            # Stop if <end> token is generated
            if predicted.item() == 1:  # Assuming 1 is the index for <end>
                break
        return sampled_ids
