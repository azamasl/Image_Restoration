# Image_Restoration
 Testing GMRES for the random nonlinear image restoration problem given in:
 "Preconditioned iterative methods for weighted Toeplitz least
 squares problems" by Benzi Ng, 2006


 The  Nonlinear Image Restoration is as follows:

 We are solving:

                      min_x ||b-s(Kx)||^2 + μ||x||^2
                      
 where

 s(x) = 30log(x),

 b = s(Kx^*) + η is the observed blurred and noisy image,

 K is a constant block Toeplitz matrix

 x^* is the original blurred image we want to recover,

 η is the white noise

 μ is the regularizer.

 Considering the implicit constraint  x > 0 we construct the Lagrangian as:

                      L(x,w) = ||b-s(Kx)||^2 + μ||x||^2 + w'x
