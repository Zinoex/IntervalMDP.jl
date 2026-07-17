@testitem "cuda: adapt" tags = [:cuda] begin
    using CUDA

    adaptor = IntervalMDP.CuModelAdaptor{Float64}
    @test IntervalMDP.valtype(adaptor) == Float64

    adaptor = IntervalMDP.CuModelAdaptor{Float32}
    @test IntervalMDP.valtype(adaptor) == Float32

    adaptor = IntervalMDP.CpuModelAdaptor{Float64}
    @test IntervalMDP.valtype(adaptor) == Float64

    adaptor = IntervalMDP.CpuModelAdaptor{Float32}
    @test IntervalMDP.valtype(adaptor) == Float32
end
