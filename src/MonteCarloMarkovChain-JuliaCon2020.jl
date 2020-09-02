
using LinearAlgebra, BenchmarkTools

A = [3 -2 0; 1 3 -1; 2 1 4]
b = [-3, -3, 6]

A

b

A\b

Matrix{Float64}(I, 3, 3)

M = Diagonal(A) # Tal vez esta es una mejor forma(?)

N = M - A

T = inv(M) * N

norm(T)

eigvals(T)

f = inv(M) * b

nT, mT = size(T)

S = zeros(Int64, nT)

[S[i] += 1 for i in 1:nT, j in 1:mT if T[i,j] != 0]

S

P = zeros(nT, mT)
[P[i,j] = 1/S[i] for i in 1:nT, j in 1:mT if T[i,j] != 0];

P

Pi = [1/nT for i in 1:nT]

ϵ = 0.5
δ = 0.5

N = floor((0.6745/δ)^2*((norm(f)^2)/(1-norm(T))^2)) + 1

function MCMC(N,T,f,P,ϵ)
    nT, mT = size(T)
    X = zeros(Float32, mT)
    cont1 = 0
    cont2 = 0
    for column_T in 1:nT # Separemos todo primero por columnas. Por qué por columnas(mT)?
        W_0 = 1
        R = 0 
        for chain in 1:N # Número de Cadenas de Markov
            W = W_0
            point = column_T # ---------------------------------------------------Aquí
            X[column_T] = W_0 * f[column_T] # ---------------------------------------Aquí
            while abs(W) >= ϵ
                nextpoint  = 1
                u = rand()
                while u >= sum(P[point, 1:nextpoint])
                    nextpoint += 1
                    cont1 += 1
                end
                if T[point, nextpoint] != 0 # Qué pasa cuando no se aplica esto?
                    W_new = W *(T[point, nextpoint]/P[point, nextpoint])
                    X[column_T] += W_new * f[nextpoint] # ------------------------Aquí
                    cont2 += 1
                end
                W = W_new
                point = nextpoint
            end
            R += X[column_T] # ---------------------------------------------------Aquí
        end
        println(R/N)
    end
    @show cont1, cont2
end

MCMC(7.053354e6,T,f,P,ϵ)

MCMC(70534,T,f,P,ϵ)
