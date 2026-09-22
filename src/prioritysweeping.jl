###################################
# Priority-queue sampling          #
###################################
#
# The priority-queue counterpart to `trajectorysampling.jl`, and structured the
# same way: one configurable strategy built from a product of orthogonal
# choices, rather than a family of fixed strategies.
#
# The two samplers answer the same question — which states should GSRDP relax
# next? — by opposite means. A trajectory answers it *forward*, by simulating
# from s₀ and relaxing what it walks over. A queue answers it *globally*, by
# ranking every state and relaxing the top of the ranking.
#
# Which is the better fit depends on the termination criterion, and GSRDP's
# default settles it. `restrict_to_initial` defaults to `false`
# (`specification.jl`), so `GapTerminationCriteria` stops on `maximum(gap)` over
# *every* state (`gsrdp.jl`). A trajectory only ever reaches a state by random
# walk from s₀; one it misses keeps its initial gap and the criterion never
# fires — which is why `ε > 0` is mandatory there rather than a tuning choice
# (see "Why the default ε is not zero" in `trajectorysampling.jl`), and why
# coverage through a bottleneck is arbitrarily slow. A queue ranks the widest-gap
# states first, and those are exactly the ones the stopping rule is waiting on.
#
# The correspondence with `trajectorysampling.jl` is close enough to reuse half
# of it outright:
#
#   TrajectorySampling            PrioritizedSweep
#   ------------------            ----------------
#   StateScore    f_S(s'|s,a)     StatePriority   f_P(s)
#   (no analogue)                 PropagationRule combine(f_P(sp), bonus)
#   SelectionPolicy (1 draw)      SelectionPolicy (k draws) — the same types
#   TemperatureSchedule           the same types, imported
#   TerminationRule (stop)        AdmissionRule (skip)
#   gauss_seidel / reverse        deliberately absent; see `PrioritizedSweep`
#
# NOTE: like `TrajectorySampling`, this module carries no docstring of its own
# (a module can't hold one alongside a same-named member) — see the docstring on
# `PrioritizedSweep` instead.
module PriorityQueueSampling

import ..IntervalMDP:
    IntervalMDP,
    PriorityQueueSamplingStrategy,
    SatisfactionMode,
    IntervalMode,
    Upper,
    Lower,
    StateIterator,
    compute_priority,
    sample,
    reset_sampling_strategy!,
    num_actions,
    system_property,
    convergence_eps,
    ismaximize,
    isoptimistic,
    _state_indices,
    _target_indices,
    _predecessor_index,
    _action_uncertainty,
    _omax_best_action,
    _is_source_state,
    _isoptimistic,
    _rnd_construct,
    _rnd_novelty,
    _rnd_train!,
    _rnd_calibrate!

# The configuration vocabulary that is genuinely shared with the trajectory
# sampler is imported, not restated: a `Boltzmann(GapDecayTemperature(...))`
# means the same thing on either side, and duplicating the types would make
# that silently untrue the first time one of them changed.
import ..IntervalMDP.TrajectorySampling:
    SelectionPolicy,
    EpsilonGreedy,
    Boltzmann,
    TemperatureSchedule,
    TemperatureContext,
    FixedTemperature,
    GapDecayTemperature,
    UpdateDecayTemperature,
    _temperature,
    _resolve_policy,
    _bound_values

###################################
# 1. State priorities              #
###################################
#
# `f_P(s)` — how urgent it is to relax `s`, higher first. This is the
# priority-queue analogue of `StateScore`, and the first three members are the
# same algebra with the transition weight `p(s'|s,a)` dropped: a trajectory
# scores a *candidate successor*, so it weights by the probability of getting
# there, while a queue scores a *state*, and the probability weighting moves
# into the propagation step (§2) where it belongs.
#
#     f_greedy(s)  = U(s) or L(s),  per the priority's `bound`
#     f_explore(s) = U(s) − L(s),   the value-function gap

"""
    StatePriority

`f_P(s)`, the priority a state carries in the queue — higher is relaxed sooner.
Evaluated by [`state_priority`](@ref).

A priority family is *self-correcting* if a never-relaxed state is guaranteed to
carry a near-maximal priority. [`GapPriority`](@ref) is: an unrelaxed state's
bounds are still at their initial `L = 0`, `U = 1`, so its gap is the largest
in the model and it cannot be starved. The others are not, and must be paired
with `aging > 0` on [`PrioritizedSweep`](@ref) — see the fairness argument
there.
"""
abstract type StatePriority end

"""
    state_priority(priority, s, value_function, model, spec) -> Float64

`f_P(s)` under `priority`.
"""
function state_priority end

"""
    GapPriority()

`f_P(s) = U(s) − L(s)`, the value-function gap — biases toward states whose
bounds are still furthest apart. This is `f_explore`, and it is the quantity
`GapTerminationCriteria` stops on, which makes it both the default and the only
self-correcting family here (see [`StatePriority`](@ref)).
"""
struct GapPriority <: StatePriority end

state_priority(::GapPriority, s, value_function, model, spec) =
    Float64(value_function.upper.current[s]) - Float64(value_function.lower.current[s])

"""
    BoundPriority(; bound = Upper)

`f_P(s) = U(s)` (or `L(s)`), the raw bound — biases toward states that look most
valuable. This is `f_greedy`. Not self-correcting: pair it with `aging > 0`.
"""
struct BoundPriority <: StatePriority
    bound::IntervalMode

    BoundPriority(; bound::IntervalMode = Upper) = new(bound)
end

state_priority(p::BoundPriority, s, value_function, model, spec) =
    Float64(_bound_values(p.bound, value_function)[s])

"""
    WeightedPriority(beta; bound = Upper)

`f_P(s) = f_greedy(s) + β·f_explore(s)` — the greedy term plus a bonus for
states whose bounds are still far apart. The direct port of
`TrajectorySampling.ExplorationScore`. `β ≥ 0`; `β = 0` reduces to
[`BoundPriority`](@ref), and large `β` approaches [`GapPriority`](@ref)'s
ordering.
"""
struct WeightedPriority <: StatePriority
    bound::IntervalMode
    beta::Float64

    function WeightedPriority(beta::Real; bound::IntervalMode = Upper)
        beta >= 0 || throw(ArgumentError("beta must be non-negative, got $beta"))
        return new(bound, Float64(beta))
    end
end

function state_priority(p::WeightedPriority, s, value_function, model, spec)
    U = Float64(value_function.upper.current[s])
    L = Float64(value_function.lower.current[s])
    return Float64(_bound_values(p.bound, value_function)[s]) + p.beta * (U - L)
end

"""
    ResidualPriority(; bound = Upper)

`f_P(s) = |maxₐ Q(s, a) − V(s)|` under `bound` — the Bellman residual at `s`:
how far `s` still is from satisfying its own Bellman equation, and hence how
much a backup there would actually move it. This is the classic prioritised
sweeping priority (Moore & Atkeson, 1993).

It measures something the gap does not: a state whose bounds are far apart but
which already sits at its own fixed point gains nothing from being relaxed
again, and this family scores it `0` where [`GapPriority`](@ref) would keep
scoring it highly. That also makes it *not* self-correcting — a converged state
is indistinguishable from a starved one — so pair it with `aging > 0`.

See [`bellman_residual_delta`](@ref) for the direction argument.
"""
struct ResidualPriority <: StatePriority
    bound::IntervalMode

    ResidualPriority(; bound::IntervalMode = Upper) = new(bound)
end

state_priority(p::ResidualPriority, s, value_function, model, spec) =
    _residual(p.bound, s, value_function, model, spec)

"""
    ActionUncertaintyPriority()

`f_P(s) = V^a(s)`, the action-selection uncertainty `_action_uncertainty`
computes — `U^{-a_L(s)}(s) − L(s)`, where `a_L(s)` is the action maximizing the
lower-bound Q-value at `s` and `U^{-a_L}(s)` is the best upper-bound Q-value at
`s` among the *other* actions. Biases toward states where it is still unsettled
which action is optimal. Not self-correcting: pair it with `aging > 0`.
"""
struct ActionUncertaintyPriority <: StatePriority end

state_priority(::ActionUncertaintyPriority, s, value_function, model, spec) =
    Float64(_action_uncertainty(model, s, value_function, spec))

###################################
# 1a. Priority lifecycle hooks     #
###################################
#
# Only `RNDPriority` needs these, but they hang off the priority family rather
# than off the strategy so that the strategy stays a plain configuration object.

"""
    initialize_priority_state!(priority, S, value_function, model, spec)

Called once per `solve`, before the initial full priority sweep, with every
state `S`. Defaults to a no-op; a family carrying a lazily constructed model —
[`RNDPriority`](@ref) and its novelty networks, whose input dimension is only
known once a model is in hand — builds it here so the sweep that follows can
already query it.
"""
initialize_priority_state!(::StatePriority, S, value_function, model, spec) = nothing

"""
    on_selected!(priority, states, value_function, model, spec)

Called with the states a `sample` call selected, immediately after selection and
hence immediately before GSRDP relaxes exactly those states. Defaults to a
no-op; [`RNDPriority`](@ref) uses it to train its novelty predictor on the
states that are about to be backed up.
"""
on_selected!(::StatePriority, states, value_function, model, spec) = nothing

"""
    reset_priority!(priority)

Called from `reset_sampling_strategy!`, once per `solve`. Defaults to a no-op.
"""
reset_priority!(::StatePriority) = nothing

###################################
# 1b. RND priority                 #
###################################
#
# `δ` is deliberately pluggable — the three functions below are ready-made, and
# any `(s, value_function, model, spec) -> Real` works.

"""
    _residual(bound, s, value_function, model, spec) -> Float64

`|maxₐ Q(s, a) − V(s)|` under `bound`, backing both [`ResidualPriority`](@ref)
and [`bellman_residual_delta`](@ref). `0` for an implicit sink, which has no
actions and hence no Bellman equation of its own.
"""
function _residual(bound::IntervalMode, s, value_function, model, spec)
    _is_source_state(model, s) || return 0.0

    V = _bound_values(bound, value_function)
    _, q = _omax_best_action(
        model,
        s,
        V,
        _isoptimistic(spec);
        maximize = _delta_maximize(spec),
    )
    isnothing(q) && return 0.0

    return abs(Float64(q) - Float64(V[s]))
end

_delta_maximize(::Nothing) = true
_delta_maximize(spec) = ismaximize(spec)

"""
    bellman_residual_delta(s, value_function, model, spec) -> Real

`δ(s) = |maxₐ Q_U(s, a) − U(s)|`, the Bellman residual at `s` under the upper
bound: how far `s` still is from satisfying its own Bellman equation, and hence
how much a backup there would move it. The default `δ` for
[`RNDPriority`](@ref), and the `bound = Upper` case of
[`ResidualPriority`](@ref).

Measured on the upper bound because that is the bound GSRDP lets drive action
selection; both the O-max/O-min ambiguity-set direction and the
maximize/minimize action-selection direction follow the specification
(`isoptimistic`/`Maximize`-`Minimize` respectively), defaulting to
optimistic-and-maximizing when `spec === nothing`. The direction must match
`bellman_update!`'s for `U` — this is a residual against `U`'s own equation, so
re-deriving it with a different direction would check the wrong equation.
"""
bellman_residual_delta(s, value_function, model, spec) =
    _residual(Upper, s, value_function, model, spec)

"""
    gap_delta(s, value_function, model, spec) -> Real

`δ(s) = |U(s) − L(s)|`, the value-function gap — the same measure
[`GapPriority`](@ref) ranks by, usable as a `δ` for [`RNDPriority`](@ref).
"""
gap_delta(s, value_function, model, spec) =
    abs(Float64(value_function.upper.current[s]) - Float64(value_function.lower.current[s]))

"""
    action_uncertainty_delta(s, value_function, model, spec) -> Real

`δ(s) = V^a(s)`, the action-selection uncertainty
[`ActionUncertaintyPriority`](@ref) ranks by, usable as a `δ` for
[`RNDPriority`](@ref).
"""
action_uncertainty_delta(s, value_function, model, spec) =
    Float64(_action_uncertainty(model, s, value_function, spec))

"""
    RNDPriority(; delta, lambda, hidden, output, lr, epochs, features, rng)

`f_P(s) = max(δ(s), λ · novelty(s))` — a value-function residual with a Random
Network Distillation novelty *floor*.

The model is known exactly, so `δ(s)` is trustworthy on its own and novelty must
not discount it — which is why this is a floor rather than the multiplicative
form RND takes under model uncertainty. What the floor buys is that a region
which has never been backed up still gets swept while its `δ` is artificially
small from a cold start.

`novelty(s)` is not "have I seen data here" — the model is known everywhere, so
that question is empty. It is a *generalized backup-recency* signal: the
predictor is trained on every state that gets backed up, so its error decays
across the whole neighbourhood of well-swept regions and stays high elsewhere.
That makes it a function-approximated stand-in for a per-state backup counter,
which a state space too large to enumerate cannot maintain exactly. See
`IntervalMDP.RandomNetworkDistillation`.

Train vs. evaluate follows from a backup being the grounded event here: a state
that is selected and relaxed is trained on; a predecessor being assigned a
propagated priority, and the initial seeding sweep, only evaluate. The floor is
applied at exactly those evaluation points — that is, to states that have not
been relaxed yet — and drops away once a state has been swept; see
[`_novelty_floor`](@ref), which also explains why that is what keeps the sampler
live.

# Keywords
- `delta`: `(s, value_function, model, spec) -> Real`, the `δ` above. Defaults
  to [`bellman_residual_delta`](@ref); [`gap_delta`](@ref) and
  [`action_uncertainty_delta`](@ref) are also provided.
- `lambda`: the novelty floor's weight. Novelty is normalized to start near `1`,
  so `λ` is in the same units as `δ`. `λ = 0` disables the floor and reduces
  this to plain prioritised sweeping over `δ`.
- `hidden`, `output`, `lr`, `epochs`: novelty network size, Adam learning rate,
  and gradient steps per relaxed batch.
- `features`: `s::CartesianIndex -> Vector{Float32}` state embedding. Defaults
  to normalized state-variable indices; models with a meaningful geometry should
  pass their own, since novelty generalizes exactly as far as the embedding says
  two states are alike.
- `rng`: optional RNG for network initialization, for reproducibility.
"""
struct RNDPriority <: StatePriority
    delta::Function
    lambda::Float64
    hidden::Int
    output::Int
    lr::Float64
    epochs::Int
    features::Any
    rng::Any

    # Built lazily in `initialize_priority_state!`: the networks' input
    # dimension is only known once there is a model to read a state's features
    # from.
    rnd::Base.RefValue{Any}
    # Which states have been relaxed at least once this run. Maintained here
    # rather than read off the strategy's tie-break clock, so the floor stays a
    # property of the priority family and the strategy needs no special case.
    relaxed::Base.RefValue{BitVector}

    function RNDPriority(;
        delta::Function = bellman_residual_delta,
        lambda::Real = 1.0,
        hidden::Int = 32,
        output::Int = 8,
        lr::Real = 1e-3,
        epochs::Int = 1,
        features = nothing,
        rng = nothing,
    )
        lambda >= 0 || throw(ArgumentError("lambda must be non-negative, got $lambda"))
        epochs >= 1 || throw(ArgumentError("epochs must be positive, got $epochs"))

        return new(
            delta,
            Float64(lambda),
            hidden,
            output,
            Float64(lr),
            epochs,
            features,
            rng,
            Ref{Any}(nothing),
            Ref(BitVector()),
        )
    end
end

function reset_priority!(p::RNDPriority)
    # Fresh networks per `solve`: a predictor still carrying the previous run's
    # training would report a whole region as already well-swept before this run
    # has backed up anything at all.
    p.rnd[] = nothing
    p.relaxed[] = BitVector()
    return nothing
end

function initialize_priority_state!(p::RNDPriority, S, value_function, model, spec)
    p.rnd[] = _rnd_construct(
        model;
        hidden = p.hidden,
        output = p.output,
        lr = p.lr,
        epochs = p.epochs,
        features = p.features,
        rng = p.rng,
    )
    # Calibrate before any training, so novelty starts near 1 everywhere and `λ`
    # reads as "the priority floor a never-backed-up state gets".
    _rnd_calibrate!(p.rnd[], S)
    p.relaxed[] = falses(length(S))

    return nothing
end

# A backup is the grounded event this signal tracks, so training happens on
# exactly the states GSRDP is about to relax — and nowhere else. Marking them
# relaxed here, before the next call's recompute, reproduces the old ordering in
# which `last_selected` was written before `on_selected!` fired.
function on_selected!(p::RNDPriority, states, value_function, model, spec)
    _rnd_train!(p.rnd[], states)

    relaxed = p.relaxed[]
    if !isempty(relaxed)
        L = LinearIndices(_state_indices(model))
        for s in states
            relaxed[L[s]] = true
        end
    end

    return nothing
end

"""
    _relaxed(priority, s, model) -> Bool

Whether `s` has been selected — and hence relaxed by GSRDP — at least once this
run.
"""
function _relaxed(p::RNDPriority, s, model)
    relaxed = p.relaxed[]
    isempty(relaxed) && return false
    return relaxed[LinearIndices(_state_indices(model))[s]]
end

"""
    _novelty_floor(priority, s, model) -> Float64

`λ · novelty(s)`, but only for a state that has not been relaxed yet; `0` once
it has.

This is what the floor is *for*: novelty is evaluated when a state enters the
queue — at seeding, and when a predecessor is pushed — to keep a region that has
never been swept from being ignored while its `δ` is still artificially small. A
state that has already been backed up is no longer making that claim, and
re-applying the floor to it would be reading the predictor as a statement about
the *future* rather than about coverage so far.

It is also what makes the sampler live. The novelty ordering across states is
arbitrary — it comes from a random target network — and training generalizes, so
every state's novelty decays roughly together. A floor that applied forever
would therefore freeze that arbitrary ordering into the priorities: with `k`
smaller than the number of states, the states that happen to rank lowest would
never be selected, and their bounds would never converge. Restricting the floor
to never-relaxed states means every priority eventually falls back to `δ` and
propagation, so an unswept state is guaranteed to reach the top of the queue
once the swept ones converge.
"""
function _novelty_floor(p::RNDPriority, s, model)
    (iszero(p.lambda) || _relaxed(p, s, model)) && return 0.0

    rnd = p.rnd[]
    isnothing(rnd) && throw(
        ArgumentError(
            "the novelty networks have not been built yet — `sample` builds them on " *
            "its first call, so `compute_priority` is only meaningful after that",
        ),
    )

    return p.lambda * _rnd_novelty(rnd, s)
end

state_priority(p::RNDPriority, s, value_function, model, spec) =
    max(Float64(p.delta(s, value_function, model, spec)), _novelty_floor(p, s, model))

###################################
# 2. Propagation rules             #
###################################
#
# After GSRDP relaxes `s` and its value moves by `Δ(s)`, the only states whose
# priority can have gone stale are the ones that *read* `V(s)` — its
# predecessors — and each can move by at most `maxₐ p̄(s|sp,a)·Δ(s)`. That bound
# is the `bonus`, and it is a certificate the trajectory sampler has no analogue
# for: a forward rollout knows where it went, never how much that mattered to
# everyone else.
#
# A `PropagationRule` says what to do with it. Prioritised sweeping (Moore &
# Atkeson, 1993) ranks *by* the bonus; the strategies this module replaces
# computed it and then discarded it, keeping only the set of stale states. Both
# are expressible, and `MaxPropagation` — which ranks by whichever of the two
# signals is more urgent — is the default.

"""
    PropagationRule

How a predecessor's own priority `f_P(sp)` combines with the propagated bound
`bonus = maxₛ p̄(s|sp,a)·Δ(s)` over the states `s` the previous batch relaxed.
Evaluated by [`_combine`](@ref).
"""
abstract type PropagationRule end

"""
    _combine(rule, own, bonus) -> Float64

The repaired priority for a state whose own priority is `own` and whose
propagated bound is `bonus`.
"""
function _combine end

"""
    MaxPropagation()

`max(f_P(sp), bonus)` — rank by whichever signal is more urgent. The default:
it is what makes this actual prioritised sweeping rather than a re-sorted
priority vector, and it cannot rank a state *below* what `f_P` alone would say,
so it is safe under every priority family.
"""
struct MaxPropagation <: PropagationRule end
_combine(::MaxPropagation, own, bonus) = max(own, bonus)

"""
    AdditivePropagation(lambda)

`f_P(sp) + λ·bonus` — treat the propagated bound as a bonus on top of the
state's own priority rather than an alternative to it. `λ ≥ 0`.
"""
struct AdditivePropagation <: PropagationRule
    lambda::Float64

    function AdditivePropagation(lambda::Real)
        lambda >= 0 || throw(ArgumentError("lambda must be non-negative, got $lambda"))
        return new(Float64(lambda))
    end
end
_combine(r::AdditivePropagation, own, bonus) = own + r.lambda * bonus

"""
    Recompute()

`f_P(sp)` — discard the bonus and simply recompute the state's own priority.
The propagated magnitude then only decides *which* states are recomputed, never
how they rank. This is what the strategies this module replaces did, and it is
kept so their deprecated constructors reproduce their results exactly.
"""
struct Recompute <: PropagationRule end
_combine(::Recompute, own, bonus) = own

###################################
# 3. Selection                     #
###################################
#
# `TrajectorySampling`'s policies draw one candidate from a scored list; here
# the same policies draw `k` *without replacement* from the whole state space.
# `TopK` is the new member — a queue has a natural deterministic answer that a
# rollout does not.
#
# Selection reads a single vector of comparison values, already carrying the
# aging term and with inadmissible states masked to `-Inf` (§4 and §5). A `-Inf`
# is never selected, exactly as in `_select`.

"""
    TopK()

Take the `k` highest-priority states, deterministically. The priority-queue
analogue of `EpsilonGreedy(0.0)`, and the default — but without that policy's
hazard, because a queue ranks every state rather than only the ones a rollout
happened to reach. Pair with `aging > 0` when the priority family is not
self-correcting; see [`PrioritizedSweep`](@ref).
"""
struct TopK <: SelectionPolicy end

# Gumbel(0, 1). Drawn via `-log(-log(u))` with `u` forced strictly positive:
# `rand()` can return exactly 0.0, which would give `-Inf` and silently make
# that state unselectable.
function _gumbel()
    u = rand()
    while u <= 0.0
        u = rand()
    end
    return -log(-log(u))
end

"""
    _top_k(values, tiebreak, k) -> Vector{Int}

The indices of the `k` largest entries of `values`, in descending order, with
`-Inf` entries dropped — so the result may be shorter than `k`, and empty when
nothing is selectable.

Ties go to the smallest `tiebreak`, which the caller sets to the clock tick each
state was last selected at, so a tied set rotates instead of one member of it
being reselected forever. Values are compared at six significant digits, which
makes that rotation reach near-ties as well: two states converging toward the
same asymptotic priority from a shared, still-stale neighbourhood can settle
into a permanent floating-point-scale gap that a strict comparison would read as
a real difference, starving whichever one loses it. The rounding is far below
any priority difference that means anything.

This handles ties. It does *not* handle a state that is genuinely and
permanently outranked — that is what `aging` on [`PrioritizedSweep`](@ref) is
for, and the two are complementary.
"""
function _top_k(values::Vector{Float64}, tiebreak::Vector{Int}, k::Int)
    n = length(values)
    k = min(k, n)
    k <= 0 && return Int[]

    top =
        partialsortperm(1:n, 1:k; by = i -> (-round(values[i]; sigdigits = 6), tiebreak[i]))
    return [i for i in top if isfinite(values[i])]
end

"""
    _select_k(policy, values, tiebreak, k) -> Vector{Int}

Draw `k` distinct state indices under `policy`, given the comparison vector
`values` and the tie-break vector `tiebreak` (see [`_top_k`](@ref)). States
scoring `-Inf` are never drawn.
"""
_select_k(::TopK, values::Vector{Float64}, tiebreak::Vector{Int}, k::Int) =
    _top_k(values, tiebreak, k)

function _select_k(
    policy::EpsilonGreedy,
    values::Vector{Float64},
    tiebreak::Vector{Int},
    k::Int,
)
    selected = _top_k(values, tiebreak, k)
    iszero(policy.p) && return selected

    # Each slot independently explores, mirroring `_select`'s single draw. The
    # candidate pool excludes what is already held so the batch stays distinct,
    # and excludes `-Inf` so exploration cannot pick an inadmissible state.
    taken = Set(selected)
    pool = [i for i in eachindex(values) if isfinite(values[i]) && !(i in taken)]

    # Swapping rather than overwriting keeps the batch distinct without any
    # further bookkeeping: the displaced state goes back into the pool, so it
    # can be drawn again for a later slot but never alongside itself.
    #
    # An exploring slot draws uniformly from the pool *plus the state it is
    # already holding*, which is the `length(pool) + 1`. Leaving that state out
    # would make exploration a forced swap rather than a uniform draw, and at
    # `p = 1` the last slot could then never keep its own greedy pick — it is
    # the only slot whose displaced state never gets offered to a later one.
    for slot in eachindex(selected)
        rand() < policy.p || continue
        j = rand(1:(length(pool) + 1))
        j <= length(pool) || continue   # drew the state already held
        selected[slot], pool[j] = pool[j], selected[slot]
    end

    return selected
end

# `k` draws without replacement with probability proportional to
# `exp(value / T)` — the Gumbel-top-k trick: perturbing each value by an
# independent Gumbel and taking the top `k` is exactly that distribution. At
# `k = 1` it agrees with `_select`'s single Boltzmann draw, and unlike that one
# it never exponentiates, so there is no overflow to guard against.
function _select_k(
    policy::Boltzmann{FixedTemperature},
    values::Vector{Float64},
    tiebreak::Vector{Int},
    k::Int,
)
    T = policy.T.t
    perturbed = [isfinite(v) ? v / T + _gumbel() : -Inf for v in values]
    return _top_k(perturbed, tiebreak, k)
end

# A decaying schedule is not a number until a `sample` call gives it one, so
# drawing straight from it is a programming error rather than a missing method.
_select_k(policy::Boltzmann, values::Vector{Float64}, tiebreak::Vector{Int}, k::Int) =
    throw(
        ArgumentError(
            "a Boltzmann policy carrying a $(typeof(policy.T)) has no temperature " *
            "until it is resolved against a TemperatureContext (`_resolve_policy`)",
        ),
    )

###################################
# 4. Admission rules               #
###################################
#
# The queue analogue of `TerminationRule`. A trajectory *stops*; a queue has
# nowhere to stop, so the corresponding choice is which states it declines to
# spend a backup on.
#
# Skipping is never permanent. A skipped state keeps its entry and its priority;
# the moment propagation raises it above the rule's threshold again it is back
# in contention. That is what makes pruning safe here and a one-way decision in
# a sampler that had to commit.

"""
    AdmissionRule

An extra condition on whether a state may be selected at all, checked by
[`admit`](@ref). A strategy carries a list of them and declines a state as soon
as any one rejects it.
"""
abstract type AdmissionRule end

"""
    admit(rule, s, value_function, model, spec) -> Bool

Whether `s` may be selected. `true` keeps it in contention.
"""
admit(::AdmissionRule, s, value_function, model, spec) = true

"""
    ConvergedSkip(eps = nothing)

Decline a state whose bounds have already met: `U(s) − L(s) < eps`. `nothing` —
the default — means `convergence_eps(prop)/2`, i.e. half the tolerance
`GapTerminationCriteria` stops on, so a declined state is already comfortably
inside the tolerance the solve is waiting for.

The test is on the **gap specifically, not on `f_P`**. Pruning on the priority
would let [`ResidualPriority`](@ref) decline a state with a wide gap that merely
happens to sit at its own fixed point — a state the termination criterion is
still waiting on. Every priority family is therefore pruned by the same
criterion the solve actually stops on.

Declining every state is safe rather than a stall: it can only happen once every
gap is below `eps`, at which point `maximum(gap) < eps ≤ convergence_eps` and
GSRDP's own criterion fires on the next check.
"""
struct ConvergedSkip <: AdmissionRule
    eps::Union{Float64, Nothing}

    function ConvergedSkip(eps::Union{Real, Nothing} = nothing)
        (eps === nothing || eps >= 0) ||
            throw(ArgumentError("eps must be non-negative, got $eps"))
        return new(eps === nothing ? nothing : Float64(eps))
    end
end

# Without a `Specification` there is no tolerance to derive, so the rule admits
# everything — the unit tests drive these helpers with `spec = nothing`.
_skip_threshold(rule::ConvergedSkip, ::Nothing) = rule.eps === nothing ? 0.0 : rule.eps
_skip_threshold(rule::ConvergedSkip, spec) =
    rule.eps === nothing ? Float64(convergence_eps(system_property(spec))) / 2 : rule.eps

function admit(rule::ConvergedSkip, s, value_function, model, spec)
    gap =
        Float64(value_function.upper.current[s]) - Float64(value_function.lower.current[s])
    return gap >= _skip_threshold(rule, spec)
end

"""
    PredicateSkip(f)

Decline `s` when `f(s, value_function, model, spec)` returns `true` — the
mirror of `TrajectorySampling.PredicateStop`, for conditions the built-in rules
do not cover.
"""
struct PredicateSkip <: AdmissionRule
    f::Function
end

admit(rule::PredicateSkip, s, value_function, model, spec) =
    !rule.f(s, value_function, model, spec)

###################################
# 5. PrioritizedSweep              #
###################################

"""
    PrioritizedSweep(; priority, propagate, policy, admit, aging, k)

Priority-queue sampling for
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref): every state carries
a priority, the top `k` are relaxed each iteration, and after each batch only
the priorities that batch could have made stale are recomputed.

# Keywords
- `priority::StatePriority = GapPriority()`: `f_P(s)`.
- `propagate::PropagationRule = MaxPropagation()`: how a predecessor's own
  priority combines with the propagated bound.
- `policy::SelectionPolicy = TopK()`: how the `k` states are drawn.
  [`TopK`](@ref), or `TrajectorySampling`'s `EpsilonGreedy` / `Boltzmann`.
- `admit = [ConvergedSkip()]`: [`AdmissionRule`](@ref)s; a state must pass all
  of them to be selectable.
- `aging::Float64 = 0.0`: the fairness term; see below.
- `k::Int = 1`: how many states to relax per iteration.

# How one call works

Each `sample` call is **repair, then select**.

Repair is prioritised sweeping's backward step. Relaxing `s` moved its value by
`Δ(s)` — *measured* against a snapshot taken when `s` was selected, not read off
the value function, because `_gsrdp!` calls `nextiteration!` immediately before
`sample` and by then `current` and `previous` agree. The only states that can
have gone stale are the ones that read `V(s)`, i.e. its predecessors, and each
can move by at most `maxₐ p̄(s|sp,a)·Δ(s)`. Exactly those are recomputed, and
combined with that bound by `propagate`.

A batch that moved nothing propagates nothing, and repair skips that step
outright. This is what makes the sweep contract: once a region reaches its fixed
point, its states stop pulling their predecessors back in, and the work
concentrates on the moving frontier without anything having to detect that. It
is sound for every priority family here because a family that reads a
neighbour's value also lists that neighbour as a successor, so a moving
neighbour reaches it through propagation anyway. A relaxed state's *own*
priority is refreshed either way, since being relaxed can change a priority
family's own state even when the value function does not move — which is exactly
how [`RNDPriority`](@ref)'s novelty floor drops away.

Selection then ranks every state by

    f_P(s) + aging·(clock − last_selected(s))

with inadmissible states masked out, and draws `k` under `policy`.

# Fairness, and why `aging` exists

`GapTerminationCriteria` stops on `maximum(gap)` over every state, and
`restrict_to_initial` defaults to `false`. A state that is never relaxed keeps
its initial gap, so the criterion never fires and `solve` does not return. Every
state must therefore be relaxed infinitely often — the same requirement that
forces `ε > 0` on `TrajectorySampling`, arrived at from the other direction.

A queue can guarantee it outright. Every priority family here is bounded, so a
state unselected for `Δt` ticks gains `aging·Δt` and must eventually outrank any
competitor: **with `aging > 0`, every state is relaxed infinitely often.** That
is deterministic, where `ε > 0` only makes it an expectation, and it costs
nothing — the ranking is a scan either way.

Selection separately breaks ties toward the least-recently-selected state (see
[`_top_k`](@ref)), which is what keeps a *tied* set rotating. The two are
complementary rather than redundant: the tie-break cannot help a state that is
genuinely outranked, and `aging` is what turns "outranked now" into "selected
eventually".

The default is `aging = 0.0` because the default priority does not need it:
[`GapPriority`](@ref) is self-correcting, since a never-relaxed state carries the
largest gap in the model and therefore sits at the top of the queue by
construction, and ties among equally-unrelaxed states are already rotated.
**Every other family should set `aging > 0`** — under [`BoundPriority`](@ref) or
[`ResidualPriority`](@ref) a state can be permanently dominated without ever
tying, and will then starve.

# Not ported from `TrajectorySampling`

`gauss_seidel` and `reverse` have no counterpart. Inside one `bellman_v!` call
`Vres` and `V` are separate buffers, so a sweep is always Jacobi; the trajectory
sampler gets Gauss-Seidel only by doling a single rollout out across solver
iterations. A queue re-ranks on every call, so holding a stale batch back is
strictly worse than re-selecting — and backward propagation already *is* the
goal-first ordering that `reverse` approximates once, maintained globally
instead of fixed at rollout time.

Only flat (non-factored) models are supported — `_predecessor_index` requires a
single `Marginal`.
"""
struct PrioritizedSweep{P <: StatePriority, R <: PropagationRule, S <: SelectionPolicy} <:
       PriorityQueueSamplingStrategy
    priority::P
    propagate::R
    policy::S
    admit::Vector{AdmissionRule}
    aging::Float64
    k::Int

    priorities::Base.RefValue{Vector{Float64}}
    last_selected::Base.RefValue{Vector{Int}}
    clock::Base.RefValue{Int}
    initialized::Base.RefValue{Bool}
    previous_selected::Base.RefValue{Vector{Int}}
    selected_snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}
    predecessor_index::Base.RefValue{Vector{Vector{Tuple{Int, Float64}}}}

    # Bellman updates issued so far — `n` for `UpdateDecayTemperature`. Tracked
    # here because `sample` is not handed the solver's iteration count.
    updates::Base.RefValue{Int}

    function PrioritizedSweep(;
        priority::P = GapPriority(),
        propagate::R = MaxPropagation(),
        policy::S = TopK(),
        admit = AdmissionRule[ConvergedSkip()],
        aging::Real = 0.0,
        k::Int = 1,
    ) where {P <: StatePriority, R <: PropagationRule, S <: SelectionPolicy}
        k >= 1 || throw(ArgumentError("k must be positive, got $k"))
        aging >= 0 || throw(ArgumentError("aging must be non-negative, got $aging"))

        return new{P, R, S}(
            priority,
            propagate,
            policy,
            collect(AdmissionRule, admit),
            Float64(aging),
            k,
            Ref(Float64[]),
            Ref(Int[]),
            Ref(0),
            Ref(false),
            Ref(Int[]),
            Ref(Tuple{Float64, Float64}[]),
            Ref(Vector{Tuple{Int, Float64}}[]),
            Ref(0),
        )
    end
end

function reset_sampling_strategy!(ss::PrioritizedSweep)
    ss.initialized[] = false
    ss.priorities[] = Float64[]
    ss.last_selected[] = Int[]
    ss.previous_selected[] = Int[]
    ss.selected_snapshot[] = Tuple{Float64, Float64}[]
    ss.predecessor_index[] = Vector{Tuple{Int, Float64}}[]
    ss.clock[] = 0
    # The temperature schedules anneal over one solve, not over the lifetime of
    # the strategy object, so the backup count restarts with it.
    ss.updates[] = 0
    reset_priority!(ss.priority)
    return nothing
end

# The generic priority entry point the parent module documents, so a
# `PrioritizedSweep` can be asked for a state's priority directly.
compute_priority(ss::PrioritizedSweep, s, value_function, model, spec) =
    state_priority(ss.priority, s, value_function, model, spec)

###################################
# 6. Sampling                      #
###################################

# The value pair snapshotted for a selected state, so that the *actual* backup
# magnitude |Δ(s)| can be measured on the next call. It has to be measured
# rather than read off the value function: `_gsrdp!` calls `nextiteration!`
# (which copies `current` into `previous`) immediately before `sample`, so by
# the time a strategy is asked for a sample the two carry identical values and
# no residual.
_snapshot(value_function, s) =
    (Float64(value_function.upper.current[s]), Float64(value_function.lower.current[s]))

_backup_magnitude(value_function, s, snapshot) = max(
    abs(Float64(value_function.upper.current[s]) - snapshot[1]),
    abs(Float64(value_function.lower.current[s]) - snapshot[2]),
)

_admissible(ss::PrioritizedSweep, s, value_function, model, spec) =
    all(r -> admit(r, s, value_function, model, spec), ss.admit)

# `Diff` for the temperature schedules. The trajectory sampler reads the gap at
# the state its rollout starts from; a queue has no such state, so it reads the
# gap the termination criterion is actually waiting on.
function _max_gap(value_function)
    U, L = value_function.upper.current, value_function.lower.current
    m = 0.0
    for i in eachindex(U)
        d = Float64(U[i]) - Float64(L[i])
        d > m && (m = d)
    end
    return m
end

# Mirrors GSRDP's own `_bellman_update_count` exactly — the sampler hands back a
# `StateIterator`, on which `project_to_state_sequence` is the identity, and
# every state in it gets a full action sweep plus the state backup.
function _count_updates!(ss::PrioritizedSweep, model, batch)
    ss.updates[] += length(batch) * (num_actions(model) + 1)
    return batch
end

function sample(ss::PrioritizedSweep, model, strategy_cache, value_function, spec)
    S = _state_indices(model)
    nS = length(S)
    ss.clock[] += 1

    if !ss.initialized[]
        ss.predecessor_index[] = _predecessor_index(model)
        initialize_priority_state!(ss.priority, S, value_function, model, spec)

        # Assigned before the sweep, not after: a priority family that depends
        # on whether a state has been relaxed yet reads it from here.
        ss.last_selected[] = zeros(Int, nS)

        priorities = Vector{Float64}(undef, nS)
        @inbounds for i in 1:nS
            priorities[i] = state_priority(ss.priority, S[i], value_function, model, spec)
        end
        ss.priorities[] = priorities
        ss.initialized[] = true
    else
        priorities = ss.priorities[]
        index = ss.predecessor_index[]
        target_linear = LinearIndices(_target_indices(model))

        # Backward propagation: relaxing `s` can only move the value of a state
        # that reads `V(s)`, i.e. a predecessor of `s`, and by at most
        # `maxₐ p̄(s|sp,a)·|Δ(s)|`.
        bonus = Dict{Int, Float64}()
        snapshots = ss.selected_snapshot[]
        for (n, i) in enumerate(ss.previous_selected[])
            Δ = _backup_magnitude(value_function, S[i], snapshots[n])

            # A relaxed state's own priority is always stale, even when its
            # value did not move: a priority family may carry state of its own
            # that being relaxed changes, and `RNDPriority` does — its novelty
            # floor drops away exactly here. Seeding the key with `0.0` puts it
            # through the same combine as everything else, so a self-loop (where
            # it is also its own predecessor) keeps whichever reading is more
            # urgent.
            bonus[i] = get(bonus, i, 0.0)

            # Nothing moved, so nothing that reads `V(s)` is stale on its
            # account. This is where a converged region stops costing anything:
            # the propagation below is the part that scales with the predecessor
            # count, and it is skipped entirely.
            Δ > 0.0 || continue

            for (j, p̄) in index[target_linear[S[i]]]
                bonus[j] = max(get(bonus, j, 0.0), p̄ * Δ)
            end
        end

        for (j, m) in bonus
            own = state_priority(ss.priority, S[j], value_function, model, spec)
            priorities[j] = _combine(ss.propagate, own, m)
        end
    end

    priorities = ss.priorities[]
    last_selected = ss.last_selected[]
    clock = ss.clock[]

    # One comparison vector: the priority, plus the aging term that guarantees
    # fairness, with inadmissible states masked out of contention entirely
    # rather than filtered after selection — filtering afterwards would hand
    # GSRDP a short batch while leaving selectable states unselected.
    values = Vector{Float64}(undef, nS)
    @inbounds for i in 1:nS
        values[i] = if _admissible(ss, S[i], value_function, model, spec)
            priorities[i] + ss.aging * (clock - last_selected[i])
        else
            -Inf
        end
    end

    policy = _resolve_policy(
        ss.policy,
        TemperatureContext(_max_gap(value_function), ss.updates[]),
    )
    top = _select_k(policy, values, last_selected, ss.k)

    for i in top
        last_selected[i] = clock
    end
    ss.previous_selected[] = top
    ss.selected_snapshot[] = [_snapshot(value_function, S[i]) for i in top]

    selected = [S[i] for i in top]
    on_selected!(ss.priority, selected, value_function, model, spec)
    _count_updates!(ss, model, selected)

    return StateIterator(selected)
end

###################################
# 7. Legacy constructors           #
###################################
#
# The four fixed strategies this module replaces, each now one configuration of
# `PrioritizedSweep`, chosen to reproduce what they actually did rather than
# what the new defaults recommend. That means `Recompute()`, under which the
# propagated magnitude decided only *which* priorities were recomputed and never
# how they ranked, and no admission rule. `RNDPriority` is the exception on the
# first count: it always did rank by the propagated magnitude, so its
# constructor keeps `MaxPropagation()`.
#
# New code should build a `PrioritizedSweep` directly. `MaxPropagation()`,
# `ConvergedSkip()` and `aging > 0` are why.

"""
    GapPriorityQueueSampling(k)

`PrioritizedSweep(; priority = GapPriority(), propagate = Recompute(), k)`.

!!! compat "Deprecated"
    Construct a [`PrioritizedSweep`](@ref) directly. The default
    `MaxPropagation()` ranks by the propagated magnitude, which this
    configuration discards.
"""
GapPriorityQueueSampling(k::Int) = PrioritizedSweep(;
    priority = GapPriority(),
    propagate = Recompute(),
    admit = AdmissionRule[],
    k = k,
)

"""
    UpperBoundPriorityQueueSampling(k)

`PrioritizedSweep(; priority = BoundPriority(), propagate = Recompute(), k)`.

!!! compat "Deprecated"
    Construct a [`PrioritizedSweep`](@ref) directly. `BoundPriority` is not
    self-correcting, so this configuration can starve a dominated state; pass
    `aging > 0` to rule that out.
"""
UpperBoundPriorityQueueSampling(k::Int) = PrioritizedSweep(;
    priority = BoundPriority(),
    propagate = Recompute(),
    admit = AdmissionRule[],
    k = k,
)

"""
    ActionUncertaintyPriorityQueueSampling(k)

`PrioritizedSweep(; priority = ActionUncertaintyPriority(), propagate = Recompute(), k)`.

!!! compat "Deprecated"
    Construct a [`PrioritizedSweep`](@ref) directly. See
    [`UpperBoundPriorityQueueSampling`](@ref) on `aging`.
"""
ActionUncertaintyPriorityQueueSampling(k::Int) = PrioritizedSweep(;
    priority = ActionUncertaintyPriority(),
    propagate = Recompute(),
    admit = AdmissionRule[],
    k = k,
)

"""
    RNDPriorityQueueSampling(k; delta, lambda, hidden, output, lr, epochs, features, rng)

`PrioritizedSweep(; priority = RNDPriority(; ...), propagate = MaxPropagation(), k)`.

!!! compat "Deprecated"
    Construct a [`PrioritizedSweep`](@ref) with an [`RNDPriority`](@ref)
    directly. Note the networks now live on the priority, so what used to be
    `ss.rnd[]` is `ss.priority.rnd[]`.
"""
RNDPriorityQueueSampling(k::Int; kwargs...) = PrioritizedSweep(;
    priority = RNDPriority(; kwargs...),
    propagate = MaxPropagation(),
    admit = AdmissionRule[],
    k = k,
)

end # module PriorityQueueSampling
