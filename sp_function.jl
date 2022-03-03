## ################ Nonlinear Image Restoration ############
#See README for the details
struct NL_Image_Reconst_Problem #problem definition struct
    K::Matrix{ComplexF64} #Block Toeplitz matrix(i.e.  discrete point spread function)
    KC::Matrix{ComplexF64}#K conjugate
    b::Matrix{Float64} #observed blurred noisy image
    μ::Float64 # regulaizer parameter
end

struct NL_Image_Reconst_ObjeGrad_Struct #return struct
    K::Function
    KT::Function
    res::Function
    L::Function #Lagrangian
    ∇xL::Function
    ∇wL::Function
    #f::Function #objective function
end

function img_recon_ObjeGrad(sp::NL_Image_Reconst_Problem)
    K,KC,b,μ = sp.K,sp.KC,sp.b,sp.μ
    function K(x)
        return real(ifft(K .* fft(x)))
    end
    function KT(x)
        return real(ifft(KC .* fft(x)))
    end
    function res(x)
        return b .- 30 .* log.(K(x)) # the residual
    end
    function L(xmat,wmat) # primal and dual
      r = res(x)
      r = r[:]
      x = xmat[:]
      w = wmat[:]
      return (r'*r) + μ*(x'*x) + w'*x
    end
    function ∇xL(x,w)
      Kx = K(x)
      D = Diagonal(1 ./ Kx)
      r = b .- 30*log.(Kx)
      grad_x = -60*(KC*D)*r .+ 2*μ*x .+ w
      return grad_x[:]
    end
    function ∇wL(x,w)
        grad_w = x
        return grad_w[:]
    end
    return NL_Image_Reconst_ObjeGrad_Struct(K,KT,res,L,∇xL,∇wL)
end

## ################# compute the condtion number as the ratio of numerically nonzero singularvalues
function myCond(M)
    U,s,V = svd(M)# s is a vector and sorted descendingly: ORDER is important
    i= length(s)
    while s[i] < 1e-8
        i=i-1
    end
    s1=s[1];sn=s[i];conM =s1/sn;
end



"""
duplicate of matlab meshgrid function
"""
function meshgrid(xin, yin)
    nx = length(xin)
    ny = length(yin)
    xout = zeros(ny, nx)
    yout = zeros(ny, nx)
    for jx = 1:nx
        for ix = 1:ny
            xout[ix, jx] = xin[jx]
            yout[ix, jx] = yin[ix]
        end
    end
    return (x = xout, y = yout)
end
