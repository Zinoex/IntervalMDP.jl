
exhaustive_cartesian(mdp::FactoredRMDP) = exhaustive_cartesian(mdp, modeltype(mdp))
exhaustive_cartesian(mdp::FactoredRMDP, ::IsIMDP) = CartesianIndices(source_shape(mdp))
exhaustive_cartesian(mdp::IntervalAmbiguitySets) = CartesianIndices(source_shape(mdp))