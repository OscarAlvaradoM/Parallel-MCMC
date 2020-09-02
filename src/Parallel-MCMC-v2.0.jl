
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

[P[i,j] = sum(P[i,1:j]) for i = 1:nT, j = 1:mT if P[i,j] != 0];
#0.0 1.0 0.0
#0.5 0.0 1.0
#0.5 1.0 0.0

T

P

e = 3

u = rand(e)
Point = [1]
Nextpoint = []
#nextpoint = [u[i] for i = 1:40, while u]
for tamaño = 1:e
    np = 1
    while u[tamaño] >= P[Point[tamaño], np]
        np += 1
    end
    push!(Nextpoint, np)
    push!(Point, np)
end

var = 1 # Aquí cambiamos la posición del vector solución que queremos aproximar, ya sea X[1], X[2] o X[3]
d_W = CUDA.ones(Float64, e) # Aquí suponemos W_0 = 1.
d_point = CuArray(Point)
d_nextpoint = CuArray{Int64}(Nextpoint)
d_W_new = CuArray{Float64}(undef, e)
d_P = CuArray{Float64}(P)
d_T = CuArray{Float64}(T)
d_X = CUDA.fill(Float64(f[var]), e)
d_f = CuArray(Float64.(f))

@cuda threads = e kernel(d_W, d_W_new, d_X, d_T, d_P, d_nextpoint, d_point, d_f)

mean(Array(d_X))

function kernel(W, W_new, X, T, P, nextpoint, point, f)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    #W_new[i] = W[i] * (T[point[i], nextpoint[i]]/P[point[i], nextpoint[i]])
    W_new[i] = W[i]*T[point[i], nextpoint[i]]/P[point[i], nextpoint[i]]
    X[i] += W_new[i] * f[nextpoint[i]]
    #W[i] = W_new[i]
    return
end

for i = 1:3
    a = T[Point[i], Nextpoint[i]]/P[Point[i], Nextpoint[i]]
    println(a%1)
end

point = 2
W_0 = 1
W = W_0
X = zeros(Float32, mT)
for tamaño = 1:20
    u = rand()
    nextpoint = 1
    while u >= P[point, nextpoint]
        nextpoint += 1
    end
    W_new = W *(T[point, nextpoint]/P[point, nextpoint])
    X[1] += W_new * f[nextpoint] # ------------------------Aquí
    W = W_new
    point = nextpoint
end

X[1]


