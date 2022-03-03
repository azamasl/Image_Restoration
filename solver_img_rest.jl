
#NOTE: For the image restoration problem x(initial guess image) is a square matrix
function secant_inv_img_rest_backtrackingLS(x0, w0, obj, sp, itNum, prt, Ftol)# gamma is the fixed stepsize
    val, ngx, ngy, ngw = 0, 0, 0, 0 #norm of the gradinet for the primal, slack and dual var
    println("J-symm method with backtracking line-search")
    x = copy(x0)
    w = copy(w0)
    N = size(x, 1)

    x = x[:]
    w = w[:]

    n = N * N
    m = n
    J = [
        I(n) zeros(n, m)
        zeros(m, n) -I(m)
    ]
    H = I(2 * n)
    F = [
        obj.∇xL(x, w)
        -obj.∇wL(x, w)
    ]
    F_old = F
    normF = LinearAlgebra.norm(F)
    normFAll = [normF]
    z = [x; w]
    Zs = z
    iter = 0
    ALPHA = 0.01
    BETA = 0.5 #backtracking ls params
    while (normF > Ftol) && iter < itNum
        p = -H * F
        ###backtracking ls
        F_old = F
        gama = 1
        while (minimum(w + gama * p[n+1:n+m]) <= 0)
            gama = BETA * gama
        end
        F = [
            obj.∇xL(x + gama * p[1:n], w + gama * p[n+1:n+m])
            -obj.∇wL(x + gama * p[1:n], w + gama * p[n+1:n+m])
        ]
        while (norm(F) > (1 - ALPHA * gama) * normF)
            gama = BETA * gama
            F = [
                obj.∇xL(x + gama * p[1:n], w + gama * p[n+1:n+m])
                -obj.∇wL(x + gama * p[1:n], w + gama * p[n+1:n+m])
            ]
        end
        ###updating z
        if prt == 1
            println("t = $gama")
        end
        s = gama * p
        z = z + s
        x = z[1:n]
        w = z[n+1:n+m]
        normF = norm(F)
        append!(normFAll, normF)
        @show size(z)
        @show size(Zs)
        Zs = hcat(Zs, z)
        ### updating H
        Y = F - F_old # Y is y in my notes

        "since Ds=-stepsize*F_old we get"
        r = Y + gama * F_old
        ns2 = LinearAlgebra.norm(s)^2
        Js = J * s
        Jr = J * r
        α = (s'*Jr)[1]#α is supposed to be an scaler
        @show typeof(α)
        a = r .- α * Js / ns2
        Ha = H * a
        denom1 = ns2 + (s'*Ha)[1]
        Ainv = H - (Ha * s' * H) / denom1
        AinvJs = Ainv * Js
        denom2 = ns2 + (Jr'*AinvJs)[1]
        H = Ainv - (AinvJs * Jr' * Ainv) / denom2

        val = obj.L(x, w)
        ngx = LinearAlgebra.norm(obj.∇xL(x, w))
        ngw = LinearAlgebra.norm(obj.∇wL(x, w))
        iter = iter + 1
        if prt == 1
            println("L = $val")
            println("|∇xL| = $ngx")
            println("|∇wL| = $ngw")
            println("||F|| = $normF")
            println("#################################End of iter $iter")
        end
    end
    append!(normFAll, normF)

    return x, w, iter, normFAll, val, normF, Zs
end
