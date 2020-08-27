
using CUDA, LinearAlgebra, Statistics

A = [3 -2 0; 1 3 -1; 2 1 4]
b = [-3, -3, 6]

M = Diagonal(A) # Tal vez esta es una mejor forma(?)

N = M - A

T = inv(M) * N

f = inv(M) * b

nT, mT = size(T)

S = zeros(Int64, nT)

[S[i] += 1 for i in 1:nT, j in 1:mT if T[i,j] != 0]

P = zeros(nT, mT)
[P[i,j] = 1/S[i] for i in 1:nT, j in 1:mT if T[i,j] != 0];
#0.0  1.0  0.0
#0.5  0.0  0.5
#0.5  0.5  0.0

Pi = [1/nT for i in 1:nT];

function P_MCMC(W, point, nextpoint, u, W_new, P, T, X1, f, suma)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    while abs(W[i]) >= 0.05
        #sync_threads()
        nextpoint[i] = Int32(1)
        suma[i] = Float32(0.)
        for k in 1:size(P, 2)
            suma[i] += P[point[i], k]
            if u[i] < suma[i]
                nextpoint[i] = k
                break 
            end
        end
        #sync_threads()
        if T[point[i], nextpoint[i]] != 0
            W_new[i] = W[i] * (T[point[i], nextpoint[i]]/P[point[i], nextpoint[i]])
            X1[i] += W_new[i] * f[nextpoint[i]]
        end
        W[i] = W_new[i]
        point[i] = nextpoint[i]
        #sync_threads()
    end
    return
end

var = 3 # Aquí cambiamos la posición del vector solución que queremos aproximar, ya sea X[1], X[2] o X[3]
N = 1024 # Número de Cadenas de Markov que crearemos
d_W = CUDA.ones(Float32, N) # Aquí suponemos W_0 = 1.
d_point = CUDA.ones(Int32, N) .* var
d_nextpoint = CUDA.ones(Int32, N)
d_u = CUDA.rand(Float32, 4000)
d_W_new = CuArray{Float32}(undef, N)
d_P = CuArray(Float32.(P))
d_T = CuArray(Float32.(T))
d_X1 = CUDA.fill(Float32(f[var]), N)
d_f = CuArray(Float32.(f))
d_suma = CUDA.zeros(Float32, N) # Esta varaible es para hacer la suma que estaba dentro del segundo while, ya
                                # que no se permite recursividad (es decir, no se puede hacer 1:4 para índices)
#d_cont = CuArray{Int32}([0]); # Sólo un contador para ver cuántas cadenas de Markov se estaban utilizando para 
                              # llegar a que abs(W) fuera menor que ϵ.

@cuda threads = 1024 P_MCMC(d_W, d_point, d_nextpoint, d_u, d_W_new, d_P, d_T, d_X1, d_f, d_suma)
# Por el momento enviamos tantos hilos comos nos lo permite un bloque, pero hay que hacer una buena gestión de
# éstos para que sea más rápido.

mean(Array(d_X1))


