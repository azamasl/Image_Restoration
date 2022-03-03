using Images, Plots, TestImages, FFTW
aa= testimage("lighthouse")
F= fftshift(fft(Float64.(Gray.(aa))))
heatmap(log.(abs.(F.*F)).+1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
# dd = randn(n)
# #dd = range(1,stop=1e3,length=n)
# dmax = maximum(dd)
# dmin = minimum(dd)
# dd = dmax .* (  1 .- ((1-rcond)/(dmax-dmin)) .* (dmax .- dd))
# D = Matrix(Diagonal(dd))
# display(cond(D,2))
#
# ## Construct linear system (3.1):
# W = Matrix(Diagonal(1 ./ (dd .^ 2))) # i.e.: W = D^-2
# A = [W K;-K' mu*I(n)]
# b = [f;zeros(n)]
#
# ## load blurred image from TwIST paper (Bioucas et al 2007)
# blur_img  = load('img/cameraman_blurred.')
#
#
# ## call GMRES
# x0 = abs.(rand(n))
# y0 = -D*(K*x0)
# b = zeros(2*n)
# b = b[:,:] #convert a vector to matrix
# zlast, ress = reg_gmres(A, b, [x0;y0], 100)
#
# display(xstar)
# display(zlast[1:n])
# display(ress)
