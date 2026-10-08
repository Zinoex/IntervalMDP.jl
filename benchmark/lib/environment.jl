# Environment capture for benchmark result files (§ Benchmark Suite, "Output").

using Dates, LinearAlgebra, SHA

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Base ref of the Phase 0 baseline (origin/main at the time). The source identity
# of a run is the pair of git tree hashes of src/ and ext/ (recorded below), so it
# does not depend on which commit is checked out. Override with BENCH_BASE_REF.
const BASE_REF = get(ENV, "BENCH_BASE_REF", "20fc03b")

function _read(path, default = "UNAVAILABLE")
    try
        return strip(read(path, String))
    catch
        return default
    end
end

function _cmd(cmd; default = "UNAVAILABLE")
    try
        return strip(read(pipeline(cmd; stderr = devnull), String))
    catch
        return default
    end
end

function git_info()
    sha = _cmd(`git -C $REPO_ROOT rev-parse HEAD`)
    short = _cmd(`git -C $REPO_ROOT rev-parse --short HEAD`)
    porcelain = _cmd(`git -C $REPO_ROOT status --porcelain`; default = "")
    srcext = _cmd(`git -C $REPO_ROOT status --porcelain -- src ext`; default = "")
    branch = _cmd(`git -C $REPO_ROOT rev-parse --abbrev-ref HEAD`)
    src_tree = _cmd(`git -C $REPO_ROOT rev-parse HEAD:src`)
    ext_tree = _cmd(`git -C $REPO_ROOT rev-parse HEAD:ext`)
    base_src = _cmd(`git -C $REPO_ROOT rev-parse $(BASE_REF):src`)
    base_ext = _cmd(`git -C $REPO_ROOT rev-parse $(BASE_REF):ext`)
    return Dict(
        "base_ref" => BASE_REF,
        "src_tree" => src_tree,
        "ext_tree" => ext_tree,
        "src_ext_equal_to_base_ref" =>
            (src_tree == base_src && ext_tree == base_ext && isempty(srcext)),
        "sha" => sha,
        "short_sha" => short,
        "branch" => branch,
        "dirty" => !isempty(porcelain),
        "src_ext_dirty" => !isempty(srcext),
        "dirty_files" => isempty(porcelain) ? String[] : split(porcelain, '\n'),
    )
end

"""
    core_type(cpu)

Core type of a logical CPU on the hybrid Intel Core Ultra 7 255H, derived from
sysfs: `cpu_core` PMU → P-core; otherwise E-core, distinguished from the
low-power-island E-cores by maximum frequency.
"""
function core_type(cpu::Integer)
    pcores = _parse_cpulist(_read("/sys/devices/cpu_core/cpus", ""))
    ecores = _parse_cpulist(_read("/sys/devices/cpu_atom/cpus", ""))
    maxf =
        tryparse(Int, _read("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq", ""))
    cpu in pcores && return "P"
    cpu in ecores && return "E"
    return isnothing(maxf) ? "unknown" : "LP-E"
end

function _parse_cpulist(s)
    out = Int[]
    isempty(s) && return out
    for part in split(s, ',')
        if occursin('-', part)
            a, b = parse.(Int, split(part, '-'))
            append!(out, a:b)
        else
            push!(out, parse(Int, part))
        end
    end
    return out
end

function cpu_info()
    ncpu = Sys.CPU_THREADS
    governors = [
        _read("/sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor") for
        i in 0:(ncpu - 1)
    ]
    epp = [
        _read("/sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference") for
        i in 0:(ncpu - 1)
    ]
    maxf = [
        _read("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq") for
        i in 0:(ncpu - 1)
    ]
    ac = String[]
    for d in
        (isdir("/sys/class/power_supply") ? readdir("/sys/class/power_supply") : String[])
        if startswith(d, "AC") || startswith(d, "ADP")
            push!(ac, "$d=" * _read("/sys/class/power_supply/$d/online"))
        end
    end
    return Dict(
        "model" => Sys.cpu_info()[1].model,
        "logical_cpus" => ncpu,
        "core_types" => Dict(string(i) => core_type(i) for i in 0:(ncpu - 1)),
        "max_freq_khz" => Dict(string(i - 1) => maxf[i] for i in 1:ncpu),
        "governor" => join(unique(governors), ","),
        "energy_performance_preference" => join(unique(epp), ","),
        "platform_profile" => _read("/sys/firmware/acpi/platform_profile"),
        "intel_pstate_no_turbo" => _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "ac_power" => join(ac, ","),
        "l3_cache" => "24 MiB (lscpu)",
        "loadavg_at_start" => Sys.loadavg(),
        "total_memory_GiB" => round(Sys.total_memory() / 2^30; digits = 2),
        "free_memory_GiB_at_start" => round(Sys.free_memory() / 2^30; digits = 2),
        "kernel" => _cmd(`uname -r`),
    )
end

"""
    pin_compact!()

Pin the default-threadpool threads 1..N to logical CPUs 0..N-1 and the
interactive threads (Julia ≥ 1.12 starts the main thread in the interactive pool
when `--threads=N` with N > 1) to CPU 0, which is the CPU of default thread 1.
The main thread does the serial parts (VI loop, precomputation) while the default
threads are idle, and is idle (waiting) while they run.
"""
function pin_compact!()
    TP = Main.ThreadPinning
    TP.pinthreads(collect(0:(Threads.nthreads(:default) - 1)); threadpool = :default)
    ni = Threads.nthreads(:interactive)
    ni > 0 && TP.pinthreads(zeros(Int, ni); threadpool = :interactive)
    return nothing
end

function thread_info(pinning)
    mapping = Dict{String, Any}[]
    for pool in (:interactive, :default)
        cpuids = try
            Main.ThreadPinning.getcpuids(; threadpool = pool)
        catch
            Int[]
        end
        tids = Threads.threadpooltids(pool)
        for (i, c) in enumerate(cpuids)
            push!(
                mapping,
                Dict(
                    "threadpool" => string(pool),
                    "thread" => i <= length(tids) ? tids[i] : -1,
                    "cpu" => c,
                    "core_type" => core_type(c),
                ),
            )
        end
    end
    return Dict(
        "nthreads" => Threads.nthreads(),
        "threadpool_default" => Threads.threadpoolsize(:default),
        "threadpool_interactive" => Threads.threadpoolsize(:interactive),
        "gc_threads" => Threads.ngcthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "blas_config" => string(BLAS.get_config()),
        "pinning" => pinning,
        "main_thread_pool" => string(Threads.threadpool(1)),
        "thread_to_cpu" => mapping,
        "JULIA_EXCLUSIVE" => get(ENV, "JULIA_EXCLUSIVE", "unset"),
    )
end

function julia_info()
    opts = Base.JLOptions()
    return Dict(
        "version" => string(VERSION),
        "commit" => Base.GIT_VERSION_INFO.commit_short,
        "opt_level" => Int(opts.opt_level),
        "check_bounds" => Int(opts.check_bounds),
        "cpu_target" => unsafe_string(opts.cpu_target),
        "word_size" => Sys.WORD_SIZE,
    )
end

gpu_info_unqueried() = Dict(
    "queried" => false,
    "reason" => "CPU backend run: CUDA is not loaded so that it cannot perturb CPU timings. See the CUDA baseline file for the GPU block.",
    # Read from the kernel driver's /proc interface (no CUDA/NVML involved).
    "model" => _proc_nvidia_model(),
    "driver_version" => _proc_nvidia_driver(),
    "runtime_version" => "unavailable (CUDA not loaded on CPU runs)",
    "versioninfo_sha256" => "unavailable (CUDA not loaded on CPU runs)",
)

function _proc_nvidia_model()
    try
        for d in readdir("/proc/driver/nvidia/gpus"; join = true)
            m = match(r"Model:\s*(.+)", read(joinpath(d, "information"), String))
            isnothing(m) || return strip(m.captures[1])
        end
    catch
    end
    return "unavailable (no /proc/driver/nvidia/gpus)"
end

function _proc_nvidia_driver()
    try
        m = match(
            r"Kernel Module\s+for\s+\S+\s+(\S+)|Kernel Module\s+(\S+)",
            read("/proc/driver/nvidia/version", String),
        )
        isnothing(m) || return something(m.captures...)
    catch
    end
    return "unavailable (no /proc/driver/nvidia/version)"
end

function gpu_info_cuda(CUDA)
    io = IOBuffer()
    vi = try
        CUDA.versioninfo(io)
        String(take!(io))
    catch e
        "versioninfo failed: " * sprint(showerror, e)
    end
    dev = CUDA.device()
    return Dict(
        "queried" => true,
        "functional" => CUDA.functional(),
        "model" => CUDA.name(dev),
        "capability" => string(CUDA.capability(dev)),
        "total_memory_GiB" => round(CUDA.totalmem(dev) / 2^30; digits = 2),
        "driver_version" => string(CUDA.driver_version()),
        "runtime_version" => string(CUDA.runtime_version()),
        "versioninfo_sha256" => bytes2hex(sha256(vi)),
        "versioninfo_head" => first(split(vi, '\n'), 12),
        "nvidia_smi" => _cmd(
            `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`;
            default = "nvidia-smi failed (see probe output in REPORT.md)",
        ),
    )
end

function environment_block(; pinning, gpu)
    return Dict(
        "date_utc" => string(Dates.now(Dates.UTC)),
        "hostname" => gethostname(),
        "git" => git_info(),
        "julia" => julia_info(),
        "threads" => thread_info(pinning),
        "cpu" => cpu_info(),
        "gpu" => gpu,
    )
end
