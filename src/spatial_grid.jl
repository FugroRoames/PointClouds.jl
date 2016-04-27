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

type Voxel{T,N}
    centres::Matrix{T}            # Centre points of voxels
    indices::Vector{NTuple{N,T}}  # Voxel indices
    voxel_size::Real              # Size of each voxel
    n_voxels::Vector{Float64}     # Number of voxels in each dimension
    offset::Vector{T}             # Offset of lowest dimension point
    ind_range::Dict{NTuple{N, T}, UnitRange{Int64}}  # Point position range for voxel index
    ind_lookup::Dict              # Dictionary lookup
end

"""
    voxelize(points, voxel_size::Real [; offset = false]) -> voxels

Create a `Voxel` data structure for the `points` using the `voxel_size`.

The `points` can either be a `PointCloud` or a `Matrix{AbstractFloat}`.
If `offset = true` the `centres` are offsetted by the minimum point position.
"""
function voxelize{T <: AbstractFloat}(points::Matrix{T},
                                      voxel_size::Real;
                                      offset::Bool = false)
    ndims, npoints = size(points)
    centred_points = minimum(points, 2)
    points = points .- centred_points

    # Count number voxels in each dimension
    n_voxels = Array(Float64, ndims)
    @inbounds for i = 1:ndims
        n_voxels[i] = ceil(maximum(points[i, :]) / voxel_size)
    end

    # Get indices for voxels
    indices = get_indices(points, voxel_size) # optimies

    # Sort point indices to get easy range look up for each voxel
    permutation = sortperm(indices)
    sorted_ind = indices[permutation]
    unique_ind = unique(sorted_ind)

    # Create a dictionary assigning voxel indices to UnitRange of points
    indices_range = get_indices_range(unique_ind, sorted_ind)

    # Look up to map the UnitRange to original point indices
    ind_lookup = Dict{Int64,Int64}()
    @inbounds for i in eachindex(permutation)
        ind_lookup[i] = permutation[i]
    end

    # Centre position of each voxel
    if offset
        centres = get_centres(unique_ind, voxel_size)
    else
        centres = get_centres(unique_ind, voxel_size) .+ centred_points
    end

    return Voxel(centres, unique_ind, voxel_size, n_voxels, collect(centred_points), indices_range, ind_lookup)
end

# Voxelize point cloud
voxelize(cloud::PointCloud, voxel_size) = voxelize(destructure(cloud.positions), voxel_size)

Base.length(v::Voxel) = length(v.indices)
Base.ndims(v::Voxel) = size(v.centres,1)

# Calculate centre point of voxels
function get_centres(indices, voxel_size)
    ndims = length(indices[1])
    nvoxels = length(indices)
    # indices position (middle point of indices)
    centres = Array(eltype(eltype(indices)), ndims, nvoxels)
    @inbounds for i in 1:nvoxels
        centres[:,i] = collect(indices[i])*voxel_size - voxel_size*0.5
    end
    return centres
end

# Calculate voxel indices
function get_indices{T <: AbstractFloat}(points::Matrix{T}, voxel_size)
    ndims, npoints = size(points)
    tmp = zeros(Float64, ndims)
    indices = Array(NTuple{ndims,T}, npoints)
    @inbounds for i in 1:npoints
        @devec tmp[:] = floor(points[:,i]./voxel_size) .+ 1.0
        indices[i] = tuple(tmp...)
    end
    return indices
end

# Get the point range for each of the same voxel indices
# Assign the point ranges to voxel indices
function get_indices_range(unique_ind, sorted_ind)
    # Determine when voxel indices change (using a similar method to zero crossings)
    tmp_arr = Array(Float64, 3, length(sorted_ind))
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
    point = point .- voxels.offset # Centre into voxelized frame
    return tuple(floor(point./voxels.voxel_size) + 1.0 ...)
end

"""
Find points in a voxel dataset specified by the voxel `indices`.

If no voxel indices are given, then all voxel points are returned. Voxel indices
can either be a `Tuple` or `Vector{Tuple}`.

### Constructors
    invoxel(voxels::Voxel)
    invoxel(voxels::Voxel, index::Tuple)
    invoxel(voxels::Voxel, indices::Vector{Tuple})

### Example

```julia
voxels = voxelize(rand(3,100)*2.0, 1.0)
indices = invoxel(voxels)                      # All voxels
indices = invoxel(voxels, (1,1,1))             # Voxel index (1,1,1)
indices = invoxel(voxels, [(1,1,1), (1,2,1)])  # Voxel index (1,1,1) and (1,2,1)
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

function Base.show(io::IO, voxels::Voxel)
    println(io, typeof(voxels))
    println(io, "  Number of dimensions: ", size(voxels.centres, 1))
    println(io, "  Number of voxels in each dimension: ", voxels.n_voxels)
    println(io, "  Size of voxels: ", voxels.voxel_size)
    print(io, "  Offset: ", voxels.offset)
end

# -----------------------------------------------------------------------------

"""
    rasterize_points(points, dx::AbstractFloat)

Rasterize `points` in 2-dimensions. Returns a `Dict` that contains the indices
of all points that are in a cell. Input `points` can either be a DxN matrix or
a `PointCloud` and `dx` is the distance between rasters.
"""
function rasterize_points(points, dx::AbstractFloat)
    _, num_points = size(points)
    points = points .- minimum(points, 2) .- 1e-9*ones(3)
    nx = ceil(Int, maximum(points[1, :])/dx)
    pixels = Dict{Int, Vector{Int}}()
    for i = 1:num_points
        key = (ceil(Int, points[1, i]/dx) + floor(Int, points[2, i]/dx) *nx)
        if haskey(pixels, key)
            push!(pixels[key], i)
        else
            pixels[key] = Vector{Int}()
            push!(pixels[key], i)
        end
    end
    return pixels
end

function rasterize_points(cloud::PointCloud, dx::AbstractFloat)
    return rasterize_points(destructure(cloud.positions), dx)
end
