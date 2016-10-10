"""
A `PointCloud` is a container for points sharing a common set of per-point
attributes.  Points within the cloud can be accessed by index, and vectors of
attributes may be accessed by name.  There is one required attribute
`position`, which is used to build a datastructure for fast spatial lookup of
k-nearest neighbours, or points within a fixed radius.

### Constructors

```julia
PointCloud(positions::Vector{FixedSizeArrays.Vec{3, AbstractFloat}})
PointCloud(positions::Matrix{AbstractFloat})
```

### Example

```julia
using FixedSizeArrays
using PointClouds

# Create positions of 3D Vec coordinates
positions = [Vec(i,-i,1.0) for i = 1:10]

# Create a PointCloud from positions
cloud = PointCloud(positions)

# Create new intensity attributes
cloud[:intensity] = collect(1:10)

# Create capture time for each point sample
cloud[:time] = [time() for i = 1:10]

# Find low intensity points
low_intensity_cloud = cloud[cloud[:intensity] .< 5]

# Find vector of time stamps per point
gps_time = low_intensity_cloud[:time]

# Find points within 5 units of [1,1,1]
nearby = cloud[inrange(cloud, [1,1,1], 5.0)]
```
"""

type PointCloud{Dim,T,SIndex}
    positions::Vector{SVector{Dim,T}}
    spatial_index::SIndex
    attributes::Dict{Symbol,Vector}
end

# FIXME should they have Nullable attributes? if attribute i smissing or someting?

# Create a `PointCloud` for FixedSizeArrays Vec using a KDTree for spatial
# indexing based on `positions`.
function PointCloud{T <: AbstractFloat}(positions::Vector{SVector{3, T}})
    PointCloud(positions, KDTree(positions), Dict{Symbol,Vector}())
end

# Create a `PointCloud` from an 3xN array of points, using a KDTree for spatial
# indexing based on `positions`.
PointCloud{T <: AbstractFloat}(points::Matrix{T}) = PointCloud(convert_positions(points))

"""
    convert_positions{T <: Real}(points::Matrix{T}) -> positions::Vector{FixedSizeArrays.Vec{3, Float64}}

Convert a standard julia array of points into a FixedSizeArray
"""
function convert_positions{T <: AbstractFloat}(points::Matrix{T})
    ndims = size(points, 1)
    return [SVector{ndims,T}(points[:, i]) for i = 1:size(points, 2)]
end

function show(io::IO, cloud::PointCloud)
    print(io, "PointCloud(N=$(length(cloud)), attributes=$(keys(cloud.attributes)))")
end

#------------------------
# Container functionality acting on columns
keys(cloud::PointCloud) = keys(cloud.attributes)
haskey(cloud::PointCloud, attrname::Symbol) = haskey(cloud.attributes, attrname)

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
    pos = positions(cloud)[row_inds]
    tree = KDTree(pos)  # TODO NEED TO GET SPATIAL INDEX FROM TYPE
    attrs = Dict{Symbol,Vector}()
    for (k,v) in cloud.attributes
        attrs[k] = v[row_inds]
    end
    PointCloud{Dim,T,SIndex}(pos, KDTree(pos), attrs)
end

# Concatenate point clouds
function vcat{Dim,T,SIndex}(cloud1::PointCloud{Dim,T,SIndex}, clouds::PointCloud{Dim,T,SIndex}...)
    pos = positions(cloud1)
    attrs = deepcopy(cloud1.attributes)
    ks = Set(keys(attrs))
    for cloud in clouds
        pos = [pos; positions(cloud)]
        ki = Set(keys(cloud))
        ki == ks || error("Cannot join clouds with inconsistent attributes")
        for (k,v) in attrs
            append!(v, cloud.attributes[k])
        end
    end
    spatial_index = KDTree(pos)  # TODO NEED TO GET SPATIAL INDEX FROM TYPE
    PointCloud{Dim,T,SIndex}(pos, spatial_index, attrs)
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
function pca_normals!{D, T}(normals, query_points::Vector{SVector{D,T}}, cloud; neighbours::Int=10)
    cov_tmp = MMatrix{3, 3, T}()
    for i = 1:length(query_points)
        @inbounds (inds, _) = knn(cloud, query_points[i], neighbours)
        isempty(inds) && continue
        fill!(cov_tmp, zero(T))
        for j = 1:length(inds)
            @inbounds p = query_points[inds[j]] - query_points[i]
            cov_tmp .+= p * p'
        end
        # Select evec of smallest eval
        evals, evecs = eig(Symmetric(cov_tmp), 1:1)
        normals[i] = evecs
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
    normals = similar(query_points, SVector{Dim,Float32})
    pca_normals!(normals, query_points, cloud; kwargs...)
    cloud[:normals] = normals
    return cloud
end
