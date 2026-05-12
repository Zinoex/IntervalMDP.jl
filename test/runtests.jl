using TestItemRunner

# `@run_package_tests` walks the test directory, picks up every
# `@testitem` block, and runs them inline (no worker pool). Per project
# convention (CLAUDE.md, plan §3 / §5) we don't want the test runner to
# greedily fork workers — VI is CPU- and memory-heavy, parallelism is
# opt-in elsewhere. A user can pass `filter = ti -> :base in ti.tags`
# (or any other predicate) to scope a run to a particular suite.
@run_package_tests
