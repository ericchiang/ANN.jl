
using Distributions

type NeuralLayer
    w::Matrix{Float64}   # weights
    b::Vector{Float64}   # biases
    a_func::Function     # activation function
    a_derv::Function     # activation funciton derivative

    # The following must be tracked for back propigation
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
    NeuralLayer(rand(Uniform(-b,b),out_dim,in_dim),
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

type ArtificalNeuralNetwork
    layers::Vector{NeuralLayer}
    n_classes::Int64
    n_layers::Int64
end

function softmax(ann_output::Vector{Float64})
    # Takes the output of a neural network and produces a valid 
    # probability distribution
    ann_output = exp(ann_output)
    ann_output / sum(ann_output)
end

function ArtificalNeuralNetwork(in_dim::Integer,
                                out_dim::Integer,
                                hidden_layer_size::Integer)
    hidden_layer = NeuralLayer(in_dim,hidden_layer_size)
    output_layer = NeuralLayer(hidden_layer_size,out_dim)
    output_layer.a_func = softmax
    ArtificalNeuralNetwork([hidden_layer,output_layer],
                            out_dim,2)
end

function forward_propigate(nl::NeuralLayer,x::Vector{Float64})
    nl.hx = x
    nl.pa = nl.b + nl.w * x
    nl.pr = nl.a_func(nl.pa)
end

function back_propigate(nl::NeuralLayer,output_gradient::Vector{Float64})
    nl.wgr = output_gradient * nl.hx' # compute weight gradient
    nl.bgr = output_gradient # compute bias gradient
    nl.w' * output_gradient # return gradient of level below
end


function fit!(ann::ArtificalNeuralNetwork,
              x::Matrix{Float64},
              y::Vector{Int64},
              epochs::Int64 = 5,
              alpha::Float64 = 0.1,
              lambda::Float64 = 0.1)
    n_obs, n_feats = size(x)
    layers = ann.layers
    for i = 1:n_obs
        y_hat = zeros(ann.n_classes)
        y_hat[y[i]] = 1.

        y_pred = predict(ann,x[i,:][:])
        output_gradient = -(y_hat - y_pred)
        for j = ann.n_layers:-1:2
            # This returns the gradient of the j-1 layer
            next_layer_gr = back_propigate(layers[j],output_gradient)
            next_layer = layers[j-1]
            output_gradient = next_layer_gr .* next_layer.a_derv(next_layer.pa)
        end
        back_propigate(layers[1],output_gradient)

        # Compute delta and step
        for j = 1:ann.n_layers
            nl = layers[j]
            # Computer L2 weight penatly
            weight_delta = nl.wgr - lambda * (2 * nl.w)
            nl.w = alpha * weight_delta + nl.w
            nl.b = alpha * nl.b
        end
    end
end


# Predict class probabilities for a given observation
function predict(ann::ArtificalNeuralNetwork,x::Vector{Float64})
    for layer in ann.layers
        x = forward_propigate(layer,x)
    end
    softmax(x)
end

# Handle a Matrix input
function predict(ann::ArtificalNeuralNetwork,x::Matrix{Float64})
    n_obs,n_feats = size(x)
    y_proba = zeros((n_obs,ann.n_classes))
    for i = 1:n_obs
        y_proba[i,:] = predict(ann,x[i,:][:])
    end
    y_proba
end
