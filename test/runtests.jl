using PointClouds
using FixedSizeArrays

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

include("cloud_test.jl")
include("spatial_grid_test.jl")
