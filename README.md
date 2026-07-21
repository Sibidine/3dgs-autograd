# Flow Visualizer for Autograd in 3D Gaussian Splatting

This implements a naive backpropagation step in 3D Gaussian Splatting, where the loss flows through the rasterizer and the projection and transformation functions. 

The `tensor.py` file builds on [Karpathy's micrograd](https://github.com/karpathy/micrograd), with the operations on scalars replaced by operations on numpy tensors. Further operations required in Gaussian splatting are also included, such as sigmoid, ReLU, matrix multiplication, swapping axes and so on. 

The Gaussian Splatting code is a recreation of my own 3D Gaussian Splatting implementation, with pytorch calls swapped out for `Tensor` ones. 

![[gaussian_splatting_forward_graph.png]]

A good few features are missing here, such as SSIM (part of the loss in the original 3DGS paper), fused CUDA kernels (that would facliltate parallelization) Adam with per-group LRs, momentum, bias correction, and Adaptive Density Control. The core idea of the GS code is to essentially be a driver for the backpropagation algorithm, and thus implementing its more complex parts were left out. The number of Gaussians and iterations were also kept to 1000 and 100 only respectively, due to hardware limitations (and inefficient CPU only implementation), hence the results are not as sharp as the original 3DGS code or any implementation of the same using standard autograd libraries.

Two visualizations are also produced: one showing the loss for 5 selected Gaussians out of the 1000 used here, which gives some idea about the location and significance of said Gaussian. The other is the flow visualization- this essentially plots out every `Tensor` operation and the gradient at that location after all iterations, which is quite handy when understanding how backpropagation happens in Gaussian Splatting(something I personally found unintuitive at first, since I had never dealt with backpropagation being used in a non Neural Network setting before).

![[gaussian_grad_curves.pngs]]

![[gaussian_grad_graph.svg]]

Usage (it is recommended to use a virtual env to run this):
```
pip install -r requirements.txt
python flowviz.py
```

