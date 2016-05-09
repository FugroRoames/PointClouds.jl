module PointClouds

import Base:
    show, keys, haskey,
    getindex, setindex!,
    vcat, length, endof

import NearestNeighbors:
    knn, inrange

using NearestNeighbors
using FixedSizeArrays

export PointCloud,
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
    # Rasterizer
    rasterize_points,
    # Create voxels
    Voxel,
    voxelize,
    invoxel,
    get_voxel_index

include("cloud.jl")
include("spatial_grid.jl")

end # module
