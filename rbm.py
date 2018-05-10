import torch
import numpy as np

class RBM():
    def __init__(self,
                 num_visible=28*28,
                 num_hidden=500,
                 k=2,
                 learning_rate=1e-3,
                 momentum_coefficient=0.5,
                 weight_decay=1e-4,
                 use_cuda=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

    def v_to_h(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def h_to_v(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def forward(self, input_data):
        return self.v_to_h(input_data)

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        return random_probabilities

    def contrastive_divergence(self, input_data):
        positive_hidden_probabilities = self.v_to_h(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.h_to_v(hidden_activations)
            hidden_probabilities = self.v_to_h(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_hidden_probabilities = hidden_probabilities
        negative_visible_probabilities = visible_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay

        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def train(self, train_loader, train_dataset, batch_size=64, num_epochs=3):
        for epoch in range(num_epochs):
            epoch_err = 0.0
            for batch, _ in train_loader:
                batch = batch.view(len(batch), self.num_visible)
                batch_err = self.contrastive_divergence(batch)
                epoch_err += batch_err
                print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_err))

        train_features = np.zeros((len(train_dataset), self.num_hidden))
        train_labels = np.zeros(len(train_dataset))

        for i, (batch, labels) in enumerate(train_loader):
            batch = batch.view(len(batch), self.num_visible)
            train_features[i*batch_size: i*batch_size+len(batch)] = self.v_to_h(batch).cpu().numpy()
            train_labels[i*batch_size: i*batch_size+len(batch)] = labels.numpy()

        return train_features, train_labels


    def extract_features(self, test_loader, test_dataset, batch_size=64, num_epochs=3):

        test_features = np.zeros((len(test_dataset), self.num_hidden))
        test_labels = np.zeros(len(test_dataset))

        for i, (batch, labels) in enumerate(test_loader):
            batch = batch.view(len(batch), self.num_visible)
            test_features[i*batch_size: i*batch_size+len(batch)] = self.v_to_h(batch).cpu().numpy()
            test_labels[i*batch_size: i*batch_size+len(batch)] = labels.numpy()

        return test_features, test_labels
