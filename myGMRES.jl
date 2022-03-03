using LinearAlgebra
#%----------------------------------------------------%
#%                 Standard GMRES
#%----------------------------------------------------%
function reg_gmres(A::Matrix, b::Matrix, x, max_iter)
    ts = size(A, 1)
    r = b - A * x
    r_norm = norm(r)
    r_normS = [r_norm]
    Q = r / r_norm
    b_norm = norm(b)
    sn = zeros(1, 0)
    cs = zeros(1, 0)
    e1 = zeros(max_iter + 1, 1)
    e1[1] = 1.0
    beta = r_norm * e1
    H = zeros(1, 0)
    zrec = x

    for k = 1:max_iter
        (hk, qk1) = arnoldi(A, Q, k)
        Q = [Q qk1]
        (hk, cs_k, sn_k) = apply_givens_rotation(hk, cs, sn, k)
        cs = [cs cs_k]
        sn = [sn sn_k]
        H = [[H; zeros(1, k - 1)] hk]

        beta[k+1] = -sn[k] * beta[k]                #update the residual vector
        beta[k] = cs[k] * beta[k]
        residual = abs(beta[k+1])
        r_normS = [r_normS; residual]

        λ = H \ beta[1:k+1]                        #calculate the result
        zk = x + Q[:, 1:k] * λ
        zrec = [zrec zk]

        if (residual <= 1e-6)
            break
        end
    end
    #return x, zrec, r_normS
    return x, r_normS, zrec
end

#%----------------------------------------------------%
#%               Standard Arnoldi
#%----------------------------------------------------%
function arnoldi(A::Matrix, Q::Matrix, k::Number)
    q = A * Q[:, k] #Krylov Vector
    h = zeros(k + 1, 1)
    for i = 1:k                                   #Modified Gram-Schmidt, keeping the Hessenberg matrix
        h[i] = q' * Q[:, i]
        q = q - h[i] * Q[:, i]
    end
    h[k+1] = norm(q)
    q = q / h[k+1]
    return (h, q)
end

#%---------------------------------------------------------------------%
#%                  Applying Givens Rotation to H col                  %
#%---------------------------------------------------------------------%
function apply_givens_rotation(h, cs, sn, k)
                                                   #apply for ith row
    for i = 1:k-1
        temp = cs[i] * h[i] + sn[i] * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i] = temp
    end
    (cs_k, sn_k) = givens_rotation(h[k], h[k+1])   #update the next sin cos values for rotation
    h[k] = cs_k * h[k] + sn_k * h[k+1]             #eliminate H(i + 1, i)
    h[k+1] = 0.0
    return (h, cs_k, sn_k)
end

                                                   #Calculate the Given rotation matrix
function givens_rotation(v1, v2)
    t = sqrt(v1^2 + v2^2)
    cs = v1 / t
    sn = v2 / t
    return (cs, sn)
end
