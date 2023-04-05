In the configuration file only the cnn layers are specified, the Linear ones will be created automatically with the parameters 
that appear on the original paper.

Tuples are composed of:
    (kernel size, number of filters, stride, padding)

Lists are composed of:
    [(tuple), (tuple), x], x=how many times those tuples should be repeteade in sequence.

Strings:
    "M": Stands for Maxpool
