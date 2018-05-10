import torch
import torch.nn as nn
from rbm import RBM


class DBN(torch.nn.Module):
    def __init__(self,
                 num_visible=28*28,
                 num_hidden=[500, 500],
                 use_cuda=False):
        super(DBN, self).__init__()
        self.n_layers = len(num_hidden)
        self.rbm_layers = []
        self.rbm_nodes = []

        for i in range(self.n_layers):
            if i==0:
                input_size = num_visible
            else:
                input_size = num_hidden[i-1]
            rbm = RBM(num_visible=input_size, num_hidden=num_hidden[i])
            self.rbm_layers.append(rbm)

        self.W_rec = [nn.Parameter(self.rbm_layers[i].weights.data.clone()) for i in range(self.n_layers - 1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].weights.data) for i in range(self.n_layers - 1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].hidden_bias.data.clone()) for i in range(self.n_layers - 1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].visible_bias.data) for i in range(self.n_layers - 1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].weights_momentum.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].visible_bias_momentum.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].hidden_bias_momentum.data)

        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])

    def forward(self, train_loader, train_dataset, batch_size=64, num_epochs=3):
        for i in range(len(self.rbm_layers)):
            # for batch, _ in train_loader:
            #     batch = batch.view(len(batch), self.rbm_layers[i].num_visible)
            #     v = self.rbm_layers[i].forward(batch)
            v = self.rbm_layers[i].forward(train_dataset)
        return v

    def train_static(self, train_loader, train_dataset, batch_size=64, num_epochs=3):
        for i in range(len(self.rbm_layers)):
            print("===================== Training =======================")
            print("Training the No.%s rbm layer" % (i+1))
            train_features, train_labels = self.rbm_layers[i].train(train_loader, train_dataset)

        return train_features, train_labels

    def Testing(self, test_loader, test_dataset, batch_size=64, num_epochs=3):
        for i in range(len(self.rbm_layers)):
            print("===================== Testing =======================")
            test_features, test_labels = self.rbm_layers[i].extract_features(test_loader, test_dataset)

        return test_features, test_labels
