###################################
# Random Network Distillation      #
###################################
#
# RND repurposed from novelty detection to a *generalized backup-recency*
# signal for prioritised-sweeping value iteration.
#
# In the reinforcement-learning setting RND answers "have I seen real data
# here?". Here the model and reward are known exactly everywhere, so that
# question is vacuous — what RND answers instead is "has this region of the
# state space received enough VI compute yet?". The predictor's error is a
# smooth, function-approximated stand-in for an exact per-state backup
# counter, which is exactly what a state space too large to enumerate cannot
# afford to maintain explicitly. Generalising that suppression to *similar*
# states, rather than tracking each one, is the point.
#
# The train/evaluate rule inverts accordingly: a Bellman backup is the
# grounded event, so the predictor is trained on every backed-up state
# ([`_rnd_train!`](@ref)) and merely evaluated, without gradients, for states
# that are only being assigned a priority ([`_rnd_novelty`](@ref)).

"""
    RandomNetworkDistillation

A frozen random `target` network and a `predictor` network trained to match
it. Novelty at a state is the predictor's squared error there: high where
the predictor has not been trained, and — because both networks are smooth
functions of the state features — low across a *neighbourhood* of states the
predictor has been trained on.

Built by [`_rnd_construct`](@ref); queried by [`_rnd_novelty`](@ref);
trained by [`_rnd_train!`](@ref).

# Fields
- `target`: frozen random network. Never optimised.
- `predictor`: same architecture, trained towards `target`.
- `opt_state`: `Flux.setup` optimiser state for `predictor`.
- `features`: `s::CartesianIndex -> Vector{Float32}` state embedding.
- `epochs::Int`: gradient steps per [`_rnd_train!`](@ref) call.
- `scale`: normalising constant, see [`_rnd_calibrate!`](@ref).
"""
struct RandomNetworkDistillation{T, P, O, F}
    target::T
    predictor::P
    opt_state::O
    features::F
    epochs::Int
    scale::Base.RefValue{Float64}
end

"""
    _rnd_default_features(model) -> (s -> Vector{Float32})

Default state embedding: each state-variable index rescaled to `[-1, 1]`
against `source_shape(model)`, so that index-adjacent states are
feature-adjacent and the predictor generalises between them.

States in this package are bare `CartesianIndex`es with no attached
features, so this is the only embedding available generically. Models with a
meaningful geometry (a grid abstraction of a continuous system, say) should
pass their own via the strategy's `features` keyword.
"""
function _rnd_default_features(model)
    shape = source_shape(model)
    return function (s)
        t = Tuple(s)
        return Float32[
            2.0f0 * (Float32(t[i] - 1) / Float32(max(1, shape[i] - 1))) - 1.0f0 for
            i in eachindex(t)
        ]
    end
end

# `Flux`'s default initialiser, optionally driven by an explicit RNG so a
# strategy can be made reproducible without touching the global RNG.
_rnd_init(::Nothing) = Flux.glorot_uniform
_rnd_init(rng) = (dims::Integer...) -> Flux.glorot_uniform(rng, dims...)

"""
    _rnd_construct(model; hidden, output, lr, epochs, features, rng)

Build a [`RandomNetworkDistillation`](@ref) for `model`. Both networks are
`Dense(d => hidden, tanh) -> Dense(hidden => output)`, independently
initialised — the predictor starts far from the target everywhere, i.e.
every state starts maximally novel.

The input dimension `d` is read off the feature map, which is why this is
built lazily on the first `sample` rather than at strategy construction.
"""
function _rnd_construct(
    model;
    hidden::Int = 32,
    output::Int = 8,
    lr::Real = 1e-3,
    epochs::Int = 1,
    features = nothing,
    rng = nothing,
)
    φ = isnothing(features) ? _rnd_default_features(model) : features
    d = length(φ(first(_state_indices(model))))
    init = _rnd_init(rng)

    # Biases are initialised randomly rather than left at Flux's default of
    # zero: with zero biases a state whose features are all zero — the centre
    # of the default embedding — drives both networks to the same output
    # identically, so its novelty would be pinned at 0 no matter how little
    # of the space around it had actually been swept.
    net() = Flux.Chain(
        Flux.Dense(d => hidden, tanh; init = init, bias = init(hidden)),
        Flux.Dense(hidden => output; init = init, bias = init(output)),
    )
    target, predictor = net(), net()

    return RandomNetworkDistillation(
        target,
        predictor,
        Flux.setup(Flux.Adam(lr), predictor),
        φ,
        epochs,
        Ref(1.0),
    )
end

# Feature matrix (d × n) for a collection of states — the layout Flux's
# `Dense` expects, one state per column.
_rnd_batch(rnd::RandomNetworkDistillation, states) =
    reduce(hcat, (rnd.features(s) for s in states))

"""
    _rnd_raw_novelty(rnd, s) -> Float64

Mean squared prediction error at `s`, before normalisation. Evaluation only
— no gradient is taken and the predictor is left untouched.
"""
function _rnd_raw_novelty(rnd::RandomNetworkDistillation, s)
    φ = rnd.features(s)
    return Float64(Flux.mse(rnd.predictor(φ), rnd.target(φ)))
end

"""
    _rnd_novelty(rnd, s) -> Float64

Normalised novelty at `s`: [`_rnd_raw_novelty`](@ref) divided by the
calibration constant from [`_rnd_calibrate!`](@ref). Starts near `1` for an
untrained region and decays towards `0` as the region is repeatedly backed
up, which puts the `λ · novelty(s)` priority floor in the same units as
`δ(s)`.
"""
_rnd_novelty(rnd::RandomNetworkDistillation, s) = _rnd_raw_novelty(rnd, s) / rnd.scale[]

"""
    _rnd_calibrate!(rnd, states)

Set the normalising constant to the mean raw novelty over `states`, measured
before any training. Called once, over the states seeded into the priority
queue.
"""
function _rnd_calibrate!(rnd::RandomNetworkDistillation, states)
    isempty(states) && return rnd

    Φ = _rnd_batch(rnd, states)
    raw = Float64(Flux.mse(rnd.predictor(Φ), rnd.target(Φ)))
    # A degenerate scale (an untrained predictor that happens to match the
    # target) would turn every novelty into `Inf`/`NaN`; fall back to 1.
    rnd.scale[] = raw > 0 ? raw : 1.0

    return rnd
end

"""
    _rnd_train!(rnd, states)

Train the predictor towards the target on `states` — one batched gradient
step per `rnd.epochs`. Called with exactly the states that were backed up,
since a backup is the grounded event this signal tracks.
"""
function _rnd_train!(rnd::RandomNetworkDistillation, states)
    isempty(states) && return rnd

    Φ = _rnd_batch(rnd, states)
    target = rnd.target(Φ)   # frozen: computed once, outside the gradient
    for _ in 1:(rnd.epochs)
        grads = Flux.gradient(m -> Flux.mse(m(Φ), target), rnd.predictor)
        Flux.update!(rnd.opt_state, rnd.predictor, grads[1])
    end

    return rnd
end
