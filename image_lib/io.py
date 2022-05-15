
import numpy as np
from imageio.v3 import imread



def read(file):
    if file.endswith('.flo'):
        return readFlow(file)
    elif file.endswith('ppm'):
        return readPPM(file)
    else:
        raise Exception('Invalid Filetype {}', file)


def readFlow(file: str) -> np.ndarray:
    f = open(file, 'rb')
    header = np.fromfile(f, np.float32, count=1).squeeze()
    if header != 202021.25:
        raise Exception('Invalid .flo file {}', file)
    w = np.fromfile(f, np.int32, 1).squeeze()
    h = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, w*h*2).reshape((h, w, 2))

    return flow

def readPPM(file: str) -> np.ndarray:
    return imread(file)
