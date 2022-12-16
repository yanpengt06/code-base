"""
    model.py: define various model classes, which will be called on train/valid loop in run.py
"""
import copy

from torch import nn


class MLP(nn.Module):

    def __init__(self, layer_num, input_dim, hidden_dim, class_num):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, class_num)
        self.hidden_layers = nn.ModuleList([copy.deepcopy(self.hidden_layer) for _ in range(layer_num - 1)])

    def forward(self, inputs):
        """
        @params:
        inputs B x input_dim
        @returns:
        logits B x C
        """
        inputs = self.input_layer(inputs) # B x H
        for hidden_layer in self.hidden_layers:
            inputs = hidden_layer(inputs) # B x H
        outputs = self.output_layer(inputs) # B x C, logits
        return outputs


if __name__ == '__main__':
    pass
