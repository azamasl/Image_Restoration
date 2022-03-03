#Benzi Ng 2004 image restoration experiment

include("myGMRES.jl")
include("sp_function.jl")
include("solver_img_rest.jl")
using Images,
    SparseArrays,
    FFTW,
    Statistics,
    Plots,
    Random,
    LinearAlgebra,
    Plots,
    CPUTime,
    JLD2


## Read the image
filename = "cameraman.tif"
μ = 1e-3
α = 0.05
BSNR = 40                                                        # (dB)signal to noise ratio
TYPE = "Nonlinear Image Restoration"
ran_seed = 17
fun_num = 6
max_it = 10#.5*1e3
prt = 1 # 0 don't print grad norm at every iterations; 1, do it.
F_tol = 1e-6

#x = load("img/cameraman.tif") #.|> float64
x = Float64.(Gray.(load("img/" * filename)))
# mu=mean(x(:));
# x=x-mu;              # no need to normalize
N = size(x, 1)#N=256
center = [N / 2; N / 2]                                           # Create a Gaussian PSF(poiont spread function), with specified center,
(JJ, II) = meshgrid(1:N, 1:N);
II = II .- center[1];
JJ = JJ .- center[2];
PSF = exp.(-((II .^ 2) .+ (JJ .^ 2)) ./ 2);
PSF = PSF ./ (2 * pi);
PSF = fftshift(PSF);                                              # circularly centered
PSF = PSF / sum(sum(PSF));                                        # normalize

K = fft(PSF);
@show typeof(K)
KC = conj(K);
Kx = real(ifft(K .* fft(x)));                                     # convolve
#spy(Kx,color=:grays)
y = 30 * log.(Kx)
Py = var(y[:]);
sigma = sqrt((Py / 10^(BSNR / 10)))
η = sigma * randn(N, N)
b = y + η;                                                       # observed image

#spy(b, color=:grays)
#save("img/" * filename * "_blur_noisy.jpg", b)

Random.seed!(ran_seed);                                          # set up the random problem and the initial point
# run the created image reconstruction problem
sp = NL_Image_Reconst_Problem(K, KC, b, μ)
obj = img_recon_ObjeGrad(sp)
# varb = var(b[:]);                                             # start with the Wiener filter
# x0 = real(fft(KC ./ (abs.(KC).^2 .+ 10*sigma^2/varb) .* ifft(b)));
x0 = zeros(size(x))                                             # starting from 0 as Benzi&Ng do
w0 = abs.(randn(size(x0)))
x_sol, w_sol, iter, nfs, val, ng, ZZ =
    secant_inv_img_rest_backtrackingLS(x0, w0, obj, sp, max_it, prt, F_tol)
#@save "output/$prob_type-$ts-lssec.jld2"  x_sol y_sol w_sol iter nfs val ng ZZ
