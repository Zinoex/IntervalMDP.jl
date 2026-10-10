using TestItemRunner

# `@run_package_tests` walks the package directory, picks up every
# `@testitem` block, and runs them inline (no worker pool). Per project
# convention (CLAUDE.md, plan §3 / §5) we don't want the test runner to
# greedily fork workers — VI is CPU- and memory-heavy, parallelism is
# opt-in elsewhere. A user can pass `filter = ti -> :base in ti.tags`
# (or any other predicate) to scope a run to a particular suite.
#
# The walk covers the whole package root, so it also finds the test files of
# git worktrees checked out under `.claude/worktrees/` — other branches, whose
# tests target their own source, not this checkout's. Skip anything under a
# `.claude` directory (an ad-hoc `@run_package_tests filter = ...` needs the
# same clause).
@run_package_tests filter = ti -> !(".claude" in splitpath(ti.filename))
