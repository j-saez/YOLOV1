import ast
import torch 
import torch.nn as nn
from models.layers import CNNBlock

KERNEL_SZ = 0
OUT_CHS = 1
STRIDE = 2
PADDING = 3
TIMES_TO_REPEAT_TUPLES = 2
TUPLE1 = 0
TUPLE2 = 1

def load_conv_layers_from_configuration(in_chs: int, config_file: str) -> nn.ModuleList:
    """
    Loads the model from a configuration file.
    Inputs:
        >> in_chs: (int) Total number of input channels.
        >> config_file: (str) Path to the configuraiton file.
    Outputs:
        >> layers: (nn.ModuleList) Contains all the layers of the model
    """
    layers = nn.Sequential()
    with open(config_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = ast.literal_eval(line.rstrip('\n'))

            # tuple = (kernel size, number of filster, stride, padding)
            if isinstance(line, tuple):
                layers.append(CNNBlock(in_chs,line[OUT_CHS],kernel_size=line[KERNEL_SZ],stride=line[STRIDE],padding=line[PADDING]))
                in_chs = line[OUT_CHS]

            # list = [(tuple1), (tuple2), x] x=how many times those tuples should be repeteade in sequence.
            elif isinstance(line, list):
                for i in range(line[TIMES_TO_REPEAT_TUPLES]):
                    layers.append(nn.Sequential(
                        CNNBlock(               in_chs, line[TUPLE1][OUT_CHS], kernel_size=line[TUPLE1][KERNEL_SZ], stride=line[TUPLE1][STRIDE], padding=line[TUPLE1][PADDING]),
                        CNNBlock(line[TUPLE1][OUT_CHS], line[TUPLE2][OUT_CHS], kernel_size=line[TUPLE2][KERNEL_SZ], stride=line[TUPLE2][STRIDE], padding=line[TUPLE2][PADDING])))
                    in_chs = line[TUPLE2][OUT_CHS]

            # M == MaxPool2d
            elif isinstance(line, str):
                if line == "M":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                config_file = config_file.split('/')[-1]
                raise Exception(f'{line} cannot be interpreted from the {config_file}.')
    return layers
