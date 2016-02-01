module PointClouds

using NearestNeighbors
using FixedSizeArrays

export PointCloud,
    # Point cloud data access
    positions, normals,
    # Spatial indexing
    knn, inrange,
    # Data handling
    readdlm_points, split_cloud,
    # Adding columns
    add_normals!

import Base:
    show, keys, haskey,
    getindex, setindex!,
    vcat, length, endof

import NearestNeighbors:
    knn, inrange


"""
A `PointCloud` is a container for points sharing a common set of per-point
attributes.  Points within the cloud can be accessed by index, and vectors of
attributes may be accessed by name.  There is one required attribute
`position`, which is used to build a datastructure for fast spatial lookup of
k-nearest neighbours, or points within a fixed radius.

Example:

```
cloud = readdlm_points("myfile.txt")

# Find low intensity points
low_intensity_cloud = cloud[cloud[:intensity] .< 10]
# Find vector of time stamps per point
gps_time = low_intensity_cloud[:time]

# Find points within 15 units of [1,1,1]
nearby = cloud[inrange(cloud, [1,1,1], 15.0)]
```
"""
type PointCloud{Dim,T,SIndex}
    positions::Vector{Vec{Dim,T}}
    spatial_index::SIndex
    attributes::Dict{Symbol,Vector}
end

"""
    PointCloud(positions)

Create a new point cloud, using a KDTree for spatial indexing based on
`positions`.
"""
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


# Access to columns with "blessed" type for efficiency.
# TODO: Figure out how we can remove these, so that eg, cloud[:normal] is the
# definitive way to access normals.
# Note that the `positions()` is used rather than `position()`, to avoid a
# clash with `Base.position()` which means IO stream position... ugh.

"Return vector of point cloud positions"
positions(cloud::PointCloud)   = cloud.positions

"Return array of point cloud normals as fixed size vectors"
function normals{Dim,T,SIndex}(cloud::PointCloud{Dim,T,SIndex})
    cloud.attributes[:normal]::Vector{Vec{Dim,Float32}}
end


#--------------------------
# Container functionality acting on rows

length(cloud::PointCloud) = length(cloud.positions)
endof(cloud::PointCloud) = length(cloud.positions)

# Subset of a point cloud
function getindex{Dim,T,SIndex}(cloud::PointCloud{Dim,T,SIndex}, row_inds::AbstractVector)
    attrs = Dict{Symbol,Vector}()
    for (k,v) in cloud.attributes
        attrs[k] = v[row_inds]
    end
    pos = attrs[:position]
    PointCloud{Dim,T,SIndex}(pos, KDTree(destructure(pos)), attrs)
end

# Concatenate point clouds
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
    PointCloud{Dim,T,SIndex}(pos, KDTree(destructure(pos)), attrs)
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
        scans[id] = allpoints[id .== scanid]
    end
    scans
end


#--------------------------
# Spatial index lookup (knn and ball queries)
knn(cloud::PointCloud, points, k) = knn(cloud.spatial_index, points, k)
inrange(cloud::PointCloud, points, radius) = inrange(cloud.spatial_index, points, radius)
knn{T<:Vec}(cloud::PointCloud, points::AbstractVector{T}, k) = knn(cloud, destructure(points), k)
inrange{T<:Vec}(cloud::PointCloud, points::AbstractVector{T}, radius) = inrange(cloud, destructure(points), radius)

# Inefficient hack to allow use with FixedSizeArrays.Vec.
# TODO(chris.foster): Remove these once Base allows for FixedSizeArrays.Vec <: AbstractVector
knn(cloud::PointCloud, point::Vec, k) = knn(cloud, [point...], k)
inrange(cloud::PointCloud, point::Vec, radius) = inrange(cloud, [point...], radius)


#------------------------
# File IO
# Column names and formats for Roames point cloud text format :(
const text_column_info = Dict(
    :X =>  (Float64, :position),  # TODO: Systematically aggregate position into a vector.
    :Y =>  (Float64, :position),
    :Z =>  (Float64, :position),
    :G =>  (Float32, :ground_height),
    :A =>  (Float32, :height_above_ground),
    :r =>  (Float32, :red),   # TODO: Map color into color type or FixedSizeVector?
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
