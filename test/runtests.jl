using TestItemRunner

# Run all `@testitem`s in the package, excluding two tag groups by default:
#  - `:cuda`    GPU tests; require a functional CUDA device (and so are skipped
#               on CI without a GPU). Run them by setting the environment variable
#               `INTERVALMDP_TEST_CUDA=1`, e.g.
#                 INTERVALMDP_TEST_CUDA=1 julia --project=. -e 'using Pkg; Pkg.test()'
#               or via the VS Code test explorer.
#  - `:mixture` the mixture tests in `base/mixture.jl`; currently disabled because
#               they target an outdated API (e.g. `OrthogonalIntervalProbabilities`).
#               Remove this exclusion once they are updated.
const EXCLUDED_TAGS =
    get(ENV, "INTERVALMDP_TEST_CUDA", "0") == "1" ? (:mixture,) : (:cuda, :mixture)
@run_package_tests filter = ti -> !any(t -> t in ti.tags, EXCLUDED_TAGS)
