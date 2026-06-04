"""Hyperparameter and configuration container."""

class Config:
    """Simple config container for training and model hyperparameters."""

    def __init__(self):
        # Model
        self.vocab_size = 50257
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768

        # Training
        self.batch_size = 12
        self.block_size = 1024
        self.learning_rate = 3e-4
        self.max_steps = 200000

        # Misc
        self.device = 'cuda' if False else 'cpu'


config = Config()
