julia-ann
=========

Implementation of backpropagation artificial neural networks in Julia.

Install (from within Julia interpreter):
-----------
```julia
Pkg.clone("git@github.com:EricChiang/ANN.jl.git")
```

Usage:
----------
```julia
using ANN

n_hidden_units = 20

ann = ArtificialNeuralNetwork(n_hidden_units)

n_obs = 150
n_feats = 80

X = rand(Float64, n_obs, n_feats)
y = rand(Int64, n_obs)

fit!(ann, X, y)

n_new_obs = 60
X_new = rand(Float64, n_new_obs, n_feats)

y_pred = predict(ann, X_new)
```


TODO:
-----

* Allow users to build multilayer networks
* Accept DataFrames as inputs. `fit!` and `predict` currently require Float64 matrices and vectors. 
