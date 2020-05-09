import numpy as np

class FeedForwardBatcher:
    def __init__(self, batch_size, shuffle=True, compansate=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.compensate = compansate

    def __call__(self, X, Y, **kwargs):
        """Yields splited X, Y matrices in minibatches of given batch_size"""
        batch_size = self.batch_size
        if (batch_size is None) or (batch_size > X.shape[-1]):
            batch_size = X.shape[-1]

        if not self.compansate:
            indx = list(range(X.shape[-1]))
            if self.shuffle:
                np.random.shuffle(indx)
            for i in range(int(X.shape[-1]/batch_size)):
                pos = i*batch_size
                # Get minibatch
                X_minibatch = X[..., indx[pos:pos+batch_size]]
                Y_minibatch = Y[..., indx[pos:pos+batch_size]]
                if i == int(X.shape[-1]/batch_size) - 1:  # Get all the remaining
                    X_minibatch = X[..., indx[pos:]]
                    Y_minibatch = Y[..., indx[pos:]]
                yield X_minibatch, Y_minibatch
        else:
            class_sum = np.sum(Y, axis=1)*Y.shape[0]
            class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)
            x_probas = np.dot(class_count, Y)
            n = X.shape[-1]
            for i in range(int(n/batch_size)):
                indxs = np.random.choice(range(n), size=batch_size, replace=True, p=x_probas)
                yield X[..., indxs], Y[..., indxs]

class RnnBatcher:
    def __init__(self, seq_length):
        self.pos = 0
        self.seq_length = seq_length
        self.batch_size = 1

    def __call__(self, X, *args, **kwargs):
        self.pos = 0
        kwargs["model"].layers[0].reset_state() # TODO: Do this right
        while self.pos + self.seq_length + 1 <= X.shape[1]:
            x = X[:, self.pos:self.pos+self.seq_length]
            y = X[:, self.pos + 1:self.pos + self.seq_length + 1]
            self.pos += 1
            yield x, y
