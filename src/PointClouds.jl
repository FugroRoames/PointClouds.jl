module PointClouds

import Base:
    show, keys, haskey,
    getindex, setindex!,
    vcat, length, endof

import NearestNeighbors:
    knn, inrange

using NearestNeighbors
using FixedSizeArrays

include("cloud.jl")
include("spatial_grid.jl")

export
    PointCloud,
    # Point cloud data access
    positions,
    normals,
    # Spatial indexing
    knn,
    inrange,
    # Data handling
    split_cloud,
    # Adding columns
    add_normals!,
    # rasterier
    rasterize_points,
    rasterize_points2

end # module
