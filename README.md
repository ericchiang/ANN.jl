julia-ann
=========

Implementation of backpropagation artificial neural networks in Julia.

Install (from within Julia interpreter):
-----------
```julia
julia> Pkg.clone("git:https://github.com/EricChiang/ANN.jl.git")
```

Usage:
----------
```julia
julia> using ANN

julia> n_hidden_units = 20

julia> ann = ArtificialNeuralNetwork(n_hidden_units)

julia> n_obs = 150

julia> n_feats = 80

julia> X = Array(Float64,n_obs,n_feats)

julia> y = Array(Int64,n_obs)

julia> fit!(ann,X,y)

julia> n_new_obs = 60

julia> X_new = Array(Float64,n_new_obs,n_feats)

julia> y_pred = predict(ann,X_new)
```


TODO:
-----

* Allow users to build multilayer networks
* Accept DataFrames as inputs. `fit!` and `predict` currently require Float64 matrices and vectors. 
