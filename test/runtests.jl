using PointClouds
using FixedSizeArrays

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

include("cloud_test.jl")
# include("test_pointclouds.jl")
include("test_spatial_grid.jl")
