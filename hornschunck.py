import numpy as np
import matplotlib.pyplot as plt
from image_lib.core import image_to_numpy, display_flow_numpy
from image_lib.io import read




def optical_flow(im1, im2):

    fx, fy, ft = image_deriv(im1, im2)
    J_11, J_12, J_13, J_22, J_23, J_33 = get_motion_tensor(fx, fy, ft)


def image_deriv(im1, im2):
    x_grad1, y_grad1 = np.gradient(im1, axis=(1,2))
    x_grad2, y_grad2 = np.gradient(im2, axis=(1,2))
    x_grad = 0.5*(x_grad1+x_grad2)
    y_grad = 0.5*(y_grad1+y_grad2)

    t_grad = np.subtract(im2, im1, dim=(1,2))
    return x_grad, y_grad, t_grad

def get_motion_tensor(fx, fy, ft):
    J_11 = np.multiply(fx,fx,dim=(1,2))
    J_12 = np.multiply(fx,fy,dim=(1,2))
    J_13 = np.multiply(fx,ft,dim=(1,2))
    J_22 = np.multiply(fy,fy,dim=(1,2))
    J_23 = np.multiply(fy,ft,dim=(1,2))
    J_33 = np.multiply(ft,ft,dim=(1,2))

    return J_11, J_12, J_13, J_22, J_23, J_33


if __name__ == "__main__":
    im1 = read("datasets/FlyingChairs_release/data/00001_img1.ppm")
    im2 = read("datasets/FlyingChairs_release/data/00001_img2.ppm")

    flow = read("datasets/FlyingChairs_release/data/00001_flow.flo")

    im1 = np.moveaxis(im1, 2, 0)
    im2 = np.moveaxis(im2, 2, 0)
    flow = np.moveaxis(flow, 2, 0)

    u = np.zeros_like(im1)
    v = np.zeros_like(im2)

    optical_flow(im1, im2)



    flow = np.moveaxis(flow, 0, 2)
    display_flow_numpy(flow)