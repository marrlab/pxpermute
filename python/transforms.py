class MinMaxScaler(object):
    def __init__(self, min_in, max_in, min_out, max_out):
        self.min_in = min_in.reshape(-1, 1, 1)
        self.max_in = max_in.reshape(-1, 1, 1)
        self.min_out = min_out
        self.max_out = max_out

    def __call__(self, tensor):
        tensor_ = (tensor - self.min_in) / (self.max_in - self.min_in)
        tensor_ = tensor_ * (self.max_out - self.min_out) + self.min_out
        tensor_[tensor_ < self.min_out] = self.min_out
        tensor_[tensor_ > self.max_out] = self.max_out
        return tensor_

    def __repr__(self):
        return self.__class__.__name__ + "(min_out={0}, max_out={1})".format(
            self.min_out, self.max_out
        )
