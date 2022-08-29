import numpy as np
from random import randint


class transformer:

    def __init__(self):
        self.We_mha_Q_1 = self.initialize_weights(512, 64)
        self.We_mha_K_1 = self.initialize_weights(512, 64)
        self.We_mha_V_1 = self.initialize_weights(512, 64)

    def encoder(self):
        pass

    def decoder(self):
        pass

    def multi_attention_head(self, input_matrix):
        """ Input: Word embeddings matrix
            Output: Returns a representation of original matrix"""

        Q = np.matmul(input_matrix, self.We_mha_Q_1)
        K = np.matmul(input_matrix, self.We_mha_K_1)
        V = np.matmul(input_matrix, self.We_mha_V_1)

        normalizing_constant = np.sqrt(np.shape(self.We_mha_V_1)[1])

        return np.matmul(self.softmax(np.matmul(Q, K.T)/normalizing_constant), V)

    @staticmethod
    def initialize_weights(dim1, dim2):

        random_weights = np.array([randint(-1000, 1000)/1000 for i in range(dim1*dim2)])

        return random_weights.reshape(dim1, dim2)

    @staticmethod
    def layer_normalization(layer, stability_constant=0):
        """ Input: Matrix
            Outputs: Layer normalized matrix"""

        mean = np.mean(layer)
        var = np.var(layer)

        return (layer - mean) / np.sqrt(var - stability_constant)

    @staticmethod
    def softmax(matrix):

        normalizing_constant = np.sum(np.exp(matrix))

        return np.exp(matrix)/normalizing_constant



t1 = transformer()
m = t1.initialize_weights(2,512)
print(t1.multi_attention_head(m))
print(np.shape(t1.multi_attention_head(m)))
