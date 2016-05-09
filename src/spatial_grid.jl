"""
` Rasterize a points cloud in 2D`

### Inputs:
* `cloud::PointCloud`: A point cloud
* `dx::AbstractFloat`: cell size

### Outputs:
* `Dict`: a dictionary that contains the indices of all the points that are in a cell
"""
function rasterize_points(cloud::PointCloud, dx::AbstractFloat)
    return rasterize_points(destructure(cloud.positions), dx)
end


"""
` Rasterize a points cloud in 2D`

### Inputs:
* `points::Matrix`: D x N matrix of points
* `dx::AbstractFloat`: cell size

### Outputs:
* `Dict`: a dictionary that contains the indices of all the points that are in a cell
"""
function rasterize_points{T <: AbstractFloat}(points::Matrix{T}, dx::T)
    _, num_points = size(points)
    points = points .- minimum(points, 2) .- 1e-9*ones(3)
    nx = ceil(Int, maximum(points[1, :])/dx)
    pixels = Dict{Tuple{Int,Int}, Vector{Int}}()
    for i = 1:num_points
        key = (ceil(Int,points[1,i]/dx), ceil(Int,points[2,i]/dx))
        if haskey(pixels, key)
            push!(pixels[key], i)
        else
            pixels[key] = Vector{Int}()
            push!(pixels[key], i)
        end
    end
    return pixels
end

"""
A data structure for organising points into voxels.

See `voxelize` for creating a voxel structure and `invoxel` for querying position of points in a
`Voxel`.

### Constructor
    Voxel(centres, indices, voxel_size, offset, ind_range, ind_lookup)

### Arguments

* `centres::Matrix{AbstractFloat}` : Centre points of voxel
* `indices::Vector{Tuple}` : Voxel indices
* `voxel_size::Real` : Size of voxel
* `offset::Vector{AbstractFloat}` : Lowest coordinate in the dataset
* `ind_range::Dict` : Point position range for voxel index
* `ind_lookup::Dict` : Dictionary lookup
"""
type Voxel{T, N}
    centres::Matrix{T}
    indices::Vector{NTuple{N, Int64}}
    voxel_size::Real
    offset::Vector{T}
    ind_range::Dict{NTuple{N, Int64}, UnitRange{Int64}}
    ind_lookup::Dict
end

"""
    voxelize(points, voxel_size::Real [; offset_flag = false]) -> voxels

Create a `Voxel` data structure for the `points` using the `voxel_size`.

The `points` can either be a `PointCloud` (see `PointClouds`) or a `Matrix{AbstractFloat}`.
If `offset_flag = true` then the voxel `centres` are offset by the minimum coordinate position.
"""
function voxelize{T <: AbstractFloat}(points::Matrix{T}, voxel_size::Real; offset_flag::Bool = false)
    ndims, npoints = size(points)
    offset = minimum(points, 2)
    points = points .- offset .+ eps()

    # Get indices for voxels
    indices = get_indices(points, voxel_size)

    # Sort point indices to get easy range look up for each voxel
    permutation = sortperm(indices)
    sorted_ind = indices[permutation]
    unique_ind = unique(sorted_ind)

    # Create a dictionary assigning voxel indices to UnitRange of points
    indices_range = get_indices_range(unique_ind, sorted_ind)

    # Look up to map the UnitRange to original point indices
    ind_lookup = Dict{Int64, Int64}()
    @inbounds for i in eachindex(permutation)
        ind_lookup[i] = permutation[i]
    end

    # Centre position of each voxel
    if offset_flag
        centres = get_centres(unique_ind, voxel_size)
    else
        centres = get_centres(unique_ind, voxel_size) .+ offset
    end

    return Voxel(centres, unique_ind, voxel_size, collect(offset), indices_range, ind_lookup)
end

# Voxelize point cloud
voxelize(cloud::PointCloud, voxel_size::Real) = voxelize(destructure(cloud.positions), voxel_size)

Base.length(v::Voxel) = length(v.indices)
Base.ndims(v::Voxel) = size(v.centres, 1)

# Calculate centre point of voxels
function get_centres(indices, voxel_size)
    ndims = length(indices[1])
    nvoxels = length(indices)
    # indices position (middle point of indices)
    centres = Array(Float64, ndims, nvoxels)
    @inbounds for i in 1:nvoxels
        centres[:,i] = collect(indices[i]) * voxel_size - voxel_size * 0.5
    end
    return centres
end

# Calculate voxel indices
function get_indices{T <: AbstractFloat}(points::Matrix{T}, voxel_size)
    ndims, npoints = size(points)
    tmp = zeros(Float64, ndims)
    indices = Array(NTuple{ndims, Int64}, npoints)
    @inbounds for i in 1:npoints
        tmp[:] = ceil(Int64, points[:, i] ./ voxel_size)
        indices[i] = tuple(tmp...)
    end
    return indices
end

# Get the point range for each of the same voxel indices
# Assign the point ranges to voxel indices
function get_indices_range(unique_ind, sorted_ind)
    # Determine when voxel indices change (using a similar method to zero crossings)
    tmp_arr = Array(Float64, length(unique_ind[1]), length(sorted_ind))
    @inbounds for i = 1:length(sorted_ind)
        tmp_arr[:, i] = collect(sorted_ind[i])
    end
    zc = sum(abs(diff(tmp_arr, 2)), 1)
    zc_ind = find(zc .!= 0.0)
    push!(zc_ind, length(sorted_ind))  # Finish at last point

    start_ind = 1
    indices_range = Dict{eltype(unique_ind), UnitRange{Int64}}()
    @inbounds for (i, ind) in enumerate(zc_ind)
        indices_range[unique_ind[i]] = start_ind:ind
        start_ind = ind + 1
    end
    return indices_range
end

"""
    get_voxel_index(voxels::Voxel, point::Vector) -> voxel_index

Return a voxel `index` given a `point`.
"""
function get_voxel_index{T <: AbstractFloat}(voxels::Voxel, point::Vector{T})
    point = point .- voxels.offset .+ eps()  # Centre into voxelized frame
    return tuple(ceil(Int64, point ./ voxel_size) ...)
end

"""
Find point indices in a voxel dataset specified by the voxel `indices`.

If no voxel indices are given, then all voxel points are returned. Voxel indices
can either be a `Tuple` or `Vector{Tuple}`.

### Constructors
    invoxel(voxels::Voxel)
    invoxel(voxels::Voxel, index::Tuple)
    invoxel(voxels::Voxel, indices::Vector{Tuple})

### Example

```julia
points = rand(3, 100) * 2.0                         # Create random points
voxel_size = 1.0                                    # Define voxel size
voxels = voxelize(points, voxel_size)               # Create voxels
indices = invoxel(voxels)                           # All voxels
indices = invoxel(voxels, (1, 1, 1))                # Voxel index (1,1,1)
indices = invoxel(voxels, [(1, 1, 1), (1, 2, 1)])   # Voxel index (1,1,1) and (1,2,1)
```
"""
invoxel(voxels::Voxel) = invoxel(voxels, voxels.indices)

# Get point indices for a voxel index
function invoxel{N, T}(voxels::Voxel, voxel_index::NTuple{N,T})
    range = get(voxels.ind_range, voxel_index, 0:0)
    indices = range != 0:0 ? indices = Int64[voxels.ind_lookup[j] for j in range] : Int64[]
end

# Get point indices for vector of voxel indices
function invoxel{N, T}(voxels::Voxel, voxel_indices::Vector{NTuple{N,T}})
    num_query_voxels = length(voxel_indices)
    indices = Array(Vector{Int64}, num_query_voxels)
    for i = 1:num_query_voxels
        indices[i] = invoxel(voxels, voxel_indices[i])
    end
    return indices
end

# Show voxel to IO
function Base.show(io::IO, voxels::Voxel)
    println(io, typeof(voxels))
    println(io, "  Number of voxels: ", length(voxels))
    println(io, "  Voxel Size: ", voxels.voxel_size)
    print(io, "  Offset: ", voxels.offset)
end
