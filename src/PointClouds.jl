module PointClouds

using NearestNeighbors
using FixedSizeArrays

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
    rasterize_points

import Base:
    show, keys, haskey,
    getindex, setindex!,
    vcat, length, endof

import NearestNeighbors:
    knn, inrange

end # module
