__precompile__()

module PointClouds

using NearestNeighbors
using StaticArrays
using TypedTables

import Base:
    @pure, @propagate_inbounds, linearindexing, LinearFast

import NearestNeighbors:
    knn, inrange

export PointCloud



#=
import Base:
    show, keys, haskey,
    getindex, setindex!,
    vcat, length, endof

import NearestNeighbors:
    knn, inrange

using NearestNeighbors
using StaticArrays

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
    add_normals!

=#


include("cloud.jl")

end # module
