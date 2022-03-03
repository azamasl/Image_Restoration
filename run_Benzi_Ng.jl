include("myGMRES.jl")
## create a random problem instance
n=3
K = zeros(n,n)
for i=1:n
  for j=1:n
    K[i,j] = 1/(sqrt(abs(i-j))+1)
  end
end
#display(K)

rcond = 1e-3 # desired recirocal conditon number for D
mu = 1e-3
alpha = 0.05
xstar = abs.(rand(n))
Kx = K*xstar
f = Kx
display(xstar)
##
dd = randn(n)
#dd = range(1,stop=1e3,length=n)
dmax = maximum(dd)
dmin = minimum(dd)
dd = dmax .* (  1 .- ((1-rcond)/(dmax-dmin)) .* (dmax .- dd))
D = Matrix(Diagonal(dd))
display(cond(D,2))

## Construct linear system (3.1):
W = Matrix(Diagonal(1 ./ (dd .^ 2))) # i.e.: W = D^-2
A = [W    K
    -K' mu*I(n)]
b = [f;zeros(n)]
## call GMRES
x0 = abs.(rand(n))
y0 = D*(f-K*x0)

#~, ress,zrec = reg_gmres(K, f[:,:], x0, 100)
~, ress,zrec = reg_gmres(A, b[:,:], [x0;y0], 100)

display(zrec[:,end])
