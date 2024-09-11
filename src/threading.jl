function _threadsfortid(extid, iter, lbody, schedule)
    lidx = iter.args[1]         # index
    range = iter.args[2]
    quote
        local threadsfor_fun
        let range = $(esc(range))
            function threadsfor_fun(tid = 1; onethread = false)
                r = range # Load into local variable
                lenr = length(r)
                # divide loop iterations among threads
                if onethread
                    tid = 1
                    len, rem = lenr, 0
                else
                    len, rem = divrem(lenr, Threads.threadpoolsize())
                end
                # not enough iterations for all the threads?
                if len == 0
                    if tid > rem
                        return
                    end
                    len, rem = 1, 0
                end
                # compute this thread's iterations
                f = firstindex(r) + ((tid - 1) * len)
                l = f + len - 1
                # distribute remaining iterations evenly
                if rem > 0
                    if tid <= rem
                        f = f + (tid - 1)
                        l = l + tid
                    else
                        f = f + rem
                        l = l + rem
                    end
                end
                # run this thread's iterations
                for i in f:l
                    local $(esc(lidx)) = @inbounds r[i]
                    local $(esc(extid)) = tid
                    $(esc(lbody))
                end
            end
        end
        if $(schedule === :dynamic || schedule === :default)
            Threads.threading_run(threadsfor_fun, false)
        elseif ccall(:jl_in_threaded_region, Cint, ()) != 0 # :static
            error("`@threads :static` cannot be used concurrently or nested")
        else # :static
            Threads.threading_run(threadsfor_fun, true)
        end
        nothing
    end
end

macro threadstid(args...)
    na = length(args)
    if na == 3
        extid, sched, ex = args
        if sched isa QuoteNode
            sched = sched.value
        elseif sched isa Symbol
            # for now only allow quoted symbols
            sched = nothing
        end
        if sched !== :static && sched !== :dynamic
            throw(ArgumentError("unsupported schedule argument in @threadstid"))
        end
    elseif na == 2
        extid = args[1]
        sched = :default
        ex = args[2]
    else
        throw(ArgumentError("wrong number of arguments in @threadstid"))
    end
    if !(isa(ex, Expr) && ex.head === :for)
        throw(ArgumentError("@threadsid requires a `for` loop expression"))
    end
    if !(ex.args[1] isa Expr && ex.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @threadsid"))
    end
    return _threadsfortid(extid, ex.args[1], ex.args[2], sched)
end
