module PointClouds

using NearestNeighbors
using FixedSizeArrays

export PointCloud,
    # Point cloud data access
    positions, normals,
    # Data handling
    readdlm_points, split_cloud

import Base:
    show, length, getindex, setindex!, keys, haskey, vcat

import NearestNeighbors:
    knn


"""
A PointCloud contains information about the coordinates of the points in the
cloud, an efficient data structure for spatial search, and various attributes
which can be accessed by name.  The only required attribute is `position`,
which is used during spatial lookup.
"""
type PointCloud{Dim,T,SIndex}
    positions::Vector{Vec{Dim,T}}
    spatial_index::SIndex
    attributes::Dict{Symbol,Vector}
end

PointCloud(positions) = PointCloud(positions, KDTree(destructure(positions)),
                                   Dict{Symbol,Vector}(:position=>positions))

function show(io::IO, cloud::PointCloud)
    print(io, "PointCloud(N=$(length(cloud)), attributes=$(keys(cloud.attributes)))")
end


#------------------------
# Container functionality acting on columns
keys(cloud::PointCloud) = keys(cloud.attributes)
haskey(cloud::PointCloud, attrname::Symbol) = (attrname == :position) || haskey(cloud.attributes, attrname)

function getindex(cloud::PointCloud, attrname::Symbol)
    haskey(cloud, attrname) || error("Point cloud $cloud has no attribute $attrname")
    cloud.attributes[attrname]
end

function setindex!(cloud::PointCloud, value::Vector, attrname::Symbol)
    if length(value) != length(cloud)
        error("length($attrname) = $(length(value)) not equal to number of points = $(length(cloud))")
    end
    cloud.attributes[attrname] = value
end

"Return vector of point cloud positions"
positions(cloud::PointCloud)   = cloud.positions

"Return array of point cloud normals as fixed size vectors"
function normals{Dim,T,SIndex}(cloud::PointCloud{Dim,T,SIndex})
    cloud.attributes[:normal]::Vector{Vec{Dim,Float32}}
end


#--------------------------
# Container functionality acting on rows
length(cloud::PointCloud) = length(cloud.positions)

function vcat{Dim,T,SIndex}(cloud1::PointCloud{Dim,T,SIndex}, clouds::PointCloud{Dim,T,SIndex}...)
    attrs = deepcopy(cloud1.attributes)
    ks = Set(keys(attrs))
    for cloud in clouds
        ki = Set(keys(cloud))
        ki == ks || error("Cannot join clouds with inconsistent attributes")
        for (k,v) in attrs
            append!(v, cloud.attributes[k])
        end
    end
    pos = attrs[:position]
    out = PointCloud{Dim,T,SIndex}(pos, KDTree(destructure(pos)), attrs)
    out
end

"""
Group points into `M = unique(scanid)` point clouds, each containing points
with the same scan id.  Returns a dictionary mapping the unique scanid to each
cloud.
"""
function split_cloud(allpoints, scanid; min_points::Int=10000)
    unique_scans = unique(scanid)
    scans = Dict{eltype(unique_scans), typeof(allpoints)}()
    for id in unique_scans
        sel = id .== scanid
        pos = positions(allpoints)[sel]
        cloud = PointCloud(pos)
        for (name,val) in allpoints.attributes
            cloud[name] = val[sel]
        end
        scans[id] = cloud
    end
    scans
end


#--------------------------
# Spatial indexing functionality
knn(cloud::PointCloud, point, k) = knn(cloud.spatial_index, point, k)


#------------------------
# File IO
# Column names and formats for Roames point cloud text format :(
const text_column_info = Dict(
    :X =>  (Float64, :position),  # TODO: Systematically aggregate position into a vector.
    :Y =>  (Float64, :position),
    :Z =>  (Float64, :position),
    :G =>  (Float32, :ground_height),
    :A =>  (Float32, :height_above_ground),
    :r =>  (Float32, :red),
    :g =>  (Float32, :green),
    :b =>  (Float32, :blue),
    :T =>  (Float64, :time),
    :R =>  (Int8,    :return_number),
    :P =>  (Int8,    :number_of_returns),
    :C =>  (UInt8,   :classification),
    :I =>  (UInt16,  :intensity),
    :L =>  (UInt16,  :point_source_id),
    :S =>  (Int64,   :flight_strip_id),
    :U =>  (Int,     :cluster),
    :Nx => (Float32, :normal_x), # TODO: Aggregate any normal info into a vector
    :Ny => (Float32, :normal_y),
    :Nz => (Float32, :normal_z),
)


"""
    readdlm_points(source)

Read a set of points from the given text `source` (an IO stream or filename),
interpreting the text column headers according to the column names defined in
the Roames ExtractPoints executable, and mapping to more sensible
multi-character names for the columns.  (See the text_column_info dictionary
for mapping details.)
"""
function readdlm_points(source::IO)
    data, column_names = readdlm(source, header=true)

    npoints = size(data,1)
    xyz_names = ["X", "Y", "Z"]
    xyz_cols = Int[findfirst(column_names, n) for n in xyz_names]
    positions = Vector{Vec{3,Float64}}(npoints)
    @fslice positions[:,:] = data[:, xyz_cols]'
    cloud = PointCloud(positions)

    for (cindex, cname) in enumerate(column_names)
        if cname in xyz_names
            continue
        end
        cname = Symbol(cname)
        ctype,fullname = text_column_info[cname]
        cloud[fullname] = map(ctype, data[:,cindex])
    end
    cloud
end

readdlm_points(filename::AbstractString) = open(io->readdlm_points(io), filename, "r")


#------------------------
# Utilities for adding columns

"""
    pca_normals!(normals, query_points, cloud; neighbours::Int=10)

Compute normals of a point cloud via PCA

For each point in the 3xN array `query_points`, compute an approximate surface
normal using PCA on the local neighbourhood, using the closest `neighbours`
points.  `cloud` should support spatial lookup via the `knn()` function.
Normals are returned in the 3xN array `normals`.

TODO: Need a generalized version of this for the PCA covariance computation.
"""
function pca_normals!(normals, query_points, cloud; neighbours::Int=10)
    for i = 1:size(query_points,2)
        p = query_points[:,i]
        inds,_ = knn(cloud, p, neighbours)
        if isempty(inds)
            continue
        end
        d = (query_points[:,inds] .- p)
        # Select evec of smallest eval
        evals, evecs = eig(Symmetric(d*d'), 1:1)
        #print("$evals")
        normals[:,i] = evecs
    end
    return normals
end


"""
Add normals to point cloud in Float32 precision, compute using PCA.

See pca_normals!() for details.
"""
function add_normals!{Dim}(cloud::PointCloud{Dim}; kwargs...)
    # Use Float32 for estimated normals - heaps of precision at half the memory
    query_points = positions(cloud)
    normals = similar(query_points, Vec{Dim,Float32})
    pca_normals!(destructure(normals), destructure(query_points), cloud; kwargs...)
    cloud[:normals] = normals
    cloud
end


end # module
