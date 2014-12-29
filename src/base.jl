
type NeuralLayer
    w::Matrix{Float64}   # weights
    b::Vector{Float64}   # biases
    a_func::Function     # activation function
    a_derv::Function     # activation funciton derivative

    # The following must be tracked for back propagation
    hx::Vector{Float64}  # input values
    pa::Vector{Float64}  # pre activation values
    pr::Vector{Float64}  # predictions (activation values)
    # Gradients
    wgr::Matrix{Float64} # weight gradient
    bgr::Vector{Float64} # bias gradient
end


# sigmoidal activation function
function sigm(a::Vector{Float64})
    1. ./ (1. + exp(-a))
end

# sigmoid derivative
function sigm_derv(a::Vector{Float64})
    s_a = sigm(a)
    s_a .* (1. - s_a)
end

function NeuralLayer(in_dim::Integer,out_dim::Integer)
    # Glorot & Bengio, 2010
    b = sqrt(6) / sqrt(in_dim + out_dim)
    NeuralLayer(2b * rand(out_dim,in_dim) - b,
                zeros(out_dim),
                sigm,
                sigm_derv,

                zeros(in_dim),
                zeros(out_dim),
                zeros(out_dim),

                zeros(out_dim,in_dim),
                zeros(out_dim)
    )
end

type ArtificialNeuralNetwork
    layers::Vector{NeuralLayer}
    hidden_sizes::Vector{Int64} # Number of nodes in each hidden level
    classes::Vector{Int64}
end

function softmax(ann_output::Vector{Float64})
    # Takes the output of a neural network and produces a valid 
    # probability distribution
    ann_output = exp(ann_output)
    ann_output / sum(ann_output)
end

function ArtificialNeuralNetwork(n_hidden_units::Integer)
    ann = ArtificialNeuralNetwork(Array(NeuralLayer,0),
                                 [n_hidden_units],
                                 Array(Int64,0))
end

function forward_propagate(nl::NeuralLayer,x::Vector{Float64})
    nl.hx = x
    nl.pa = nl.b + nl.w * x
    nl.pr = nl.a_func(nl.pa)
end

function back_propagate(nl::NeuralLayer,output_gradient::Vector{Float64})
    nl.wgr = output_gradient * nl.hx' # compute weight gradient
    nl.bgr = output_gradient # compute bias gradient
    nl.w' * output_gradient # return gradient of level below
end

function init!(ann::ArtificialNeuralNetwork,
                     x::Matrix{Float64},
                     y::Vector{Int64})
    layers = Array(NeuralLayer,length(ann.hidden_sizes) + 1)
    ann.classes = unique(y)
    sort!(ann.classes)
    input_dim = size(x)[2]
    for i = 1:length(ann.hidden_sizes)
        out_dim = ann.hidden_sizes[i]
        layers[i] = NeuralLayer(input_dim,out_dim)
        input_dim = out_dim
    end
    layers[length(layers)] = NeuralLayer(input_dim,length(ann.classes))
    layers[length(layers)].a_func = softmax
    ann.layers = layers
    ann
end

function fit!(ann::ArtificialNeuralNetwork,
              x::Matrix{Float64},
              y::Vector{Int64};
              epochs::Int64 = 5,
              alpha::Float64 = 0.1,
              lambda::Float64 = 0.1)
    init!(ann,x,y)
    n_obs, n_feats = size(x)
    layers = ann.layers
    n_layers = length(layers)
    for _ = 1:epochs
        for i = 1:n_obs
            y_hat = zeros(length(ann.classes))
            y_hat[findfirst(ann.classes,y[i])] = 1.

            y_pred = predict(ann,x[i,:][:])
            output_gradient = -(y_hat - y_pred)
            for j = n_layers:-1:2
                # This returns the gradient of the j-1 layer
                next_layer_gr = back_propagate(layers[j],output_gradient)
                next_layer = layers[j-1]
                output_gradient = next_layer_gr .* next_layer.a_derv(next_layer.pa)
            end
            back_propagate(layers[1],output_gradient)

            # Compute delta and step
            for j = 1:n_layers
                nl = layers[j]
                # Computer L2 weight penatly
                weight_delta = (-nl.wgr) - (lambda * (2 * nl.w))
                nl.w = alpha * weight_delta + nl.w
                nl.b = alpha * (-nl.bgr) + nl.b
            end
        end
    end
end


# Predict class probabilities for a given observation
function predict(ann::ArtificialNeuralNetwork,x::Vector{Float64})
    for i in 1:length(ann.layers)
        x = forward_propagate(ann.layers[i],x)
    end
    x
end

# Handle a Matrix input
function predict(ann::ArtificialNeuralNetwork,x::Matrix{Float64})
    n_obs,n_feats = size(x)
    y_proba = zeros((n_obs,length(ann.classes)))
    for i = 1:n_obs
        y_proba[i,:] = predict(ann,x[i,:][:])
    end
    y_proba
end
