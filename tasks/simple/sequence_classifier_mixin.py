import torch
import torch.nn
import torch.utils.data
from models import SequenceClassifier
from interfaces import SequenceClassifierInterface
from .. import args
import framework
from layers import CudaLSTM
from layers.dnc import DNC, LSTMController, FeedforwardController
import functools


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-seq_classifier.rnn", default="lstm", choice=["lstm", "dnc", "dnc_ff"])
    parser.add_argument("-thinking_steps", default=0)


class SequenceClassifierMixin:
    def create_model(self) -> torch.nn.Module:
        rnns = {
            "lstm": CudaLSTM,
            "dnc": lambda embedding_size, hidden_size, n_layers, dropout: DNC(embedding_size, hidden_size, hidden_size,
                    21, 4, LSTMController([hidden_size]), batch_first=False),
            "dnc_ff": lambda embedding_size, hidden_size, n_layers, dropout: DNC(embedding_size, hidden_size, hidden_size,
                    21, 4, FeedforwardController(self.helper.args.layer_sizes), batch_first=False)
        }
        model = SequenceClassifier(len(self.train_set.in_vocabulary),
                                   len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                   self.helper.args.n_layers,
                                   self.helper.args.embedding_size,
                                   self.helper.args.dropout,
                                   lstm = rnns.get(self.helper.args.seq_classifier.rnn),
                                   n_thinking_steps=self.helper.args.thinking_steps)

        self.n_weights = sum(p.numel() for p in model.parameters())
        return model

    def create_model_interface(self):
        self.model_interface = SequenceClassifierInterface(self.model)
