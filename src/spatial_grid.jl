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

typealias VoxelId NTuple{3, Int}

"""
Creates a sparse spatial grid by organising 3D points into voxels for optimised lookup.

See `voxelize` for creating a sparse voxel grid, `voxelids` for obtaining voxel ids and
`invoxel` for querying points indices.

### Constructor
    SparseVoxelGrid(voxel_size, ind_range, indices)

### Arguments

* `voxel_size::AbstractFloat` : Side length of each voxel cell
* `ind_range::Dict{NTuple{3,Int}, UnitRange{Int}}` : Store a range of point indices for each unique voxel id
* `indices::Vector{Int}` : A vector holding the original point indices

### Usage

To create a spatial grid with voxel side length of 10 metres for arbitrary points:
```julia
using PointClouds
points = rand(3, 100000) * 20.0
grid = voxelize(points, 10.0)
```
To find the point indices in each voxel:
```julia
for voxel_id in voxelids(grid)
    # Create iterator over point indices
    indices_iter = invoxel(grid, voxel_id)

    # Collect all indices for current voxel
    point_inds = collect(indices_iter)

    # Or, iterate each individual point
    for point_idx in indices_iter
        # Then something with point_idx
    end
end
```
"""
type SparseVoxelGrid{T <: AbstractFloat}
    voxel_size::T
    ind_range::Dict{VoxelId, UnitRange{Int}}
    indices::Vector{Int}
end

"""
    make_voxel_id(points::Matrix, index, voxel_size)

Create a 3D voxel id tuple for the point specified by the column index.
"""
function make_voxel_id(points::Matrix, index, voxel_size)
    (ceil(Int, (points[1, index] + eps()) / voxel_size),
     ceil(Int, (points[2, index] + eps()) / voxel_size),
     ceil(Int, (points[3, index] + eps()) / voxel_size))
end

"""
    voxel_centre(voxel_size::AbstractFloat, voxel_id::VoxelId)

Calculate the centre point of each voxel in the spatial grid.
"""
function voxel_centre(voxel_size::AbstractFloat, voxel_id::VoxelId)
    centre = collect(voxel_id) * voxel_size - voxel_size * 0.5
end

"""
    voxelize(points, voxel_size::AbstractFloat) -> grid::SparseVoxelGrid

Create a `SparseVoxelGrid` data structure for the `points` using the `voxel_size`. The `points`
can be a `PointCloud` (see `PointClouds`) or a `Matrix{AbstractFloat}`.

See `SparseVoxelGrid` for detailed usage.
"""
function voxelize{T <: AbstractFloat}(points::Matrix{T}, voxel_size::T)
    ndims, npoints = size(points)
    ndims != 3 && throw(ArgumentError("Points dimensions are $(size(points)), should be a 3xN matrix."))

    # Assign each point to a voxel id
    voxel_ids = Vector{VoxelId}(npoints)
    for j = 1:npoints
        voxel_ids[j] = make_voxel_id(points, j, voxel_size)
    end

    group_counts = Dict{VoxelId, Int}()
    for id in voxel_ids
        group_counts[id] = get(group_counts, id, 0) + 1
    end

    ind_range = Dict{VoxelId, UnitRange{Int}}()
    current_index = 1
    for (group_id, group_size) in group_counts
        ind_range[group_id] = current_index:current_index+group_size-1
        current_index += group_size
    end

    index = Vector{Int}(npoints)
    for j = 1:npoints
        id = voxel_ids[j]
        index_in_group = group_counts[id]
        group_counts[id] = index_in_group - 1
        index[first(ind_range[id]) + index_in_group-1] = j
    end

    return SparseVoxelGrid(voxel_size, ind_range, index)
end

# Voxelize point cloud
voxelize(cloud::PointCloud, voxel_size::AbstractFloat) = voxelize(destructure(cloud.positions), voxel_size)

# Functionality for SparseVoxelGrid
Base.length(grid::SparseVoxelGrid) = length(grid.ind_range)
Base.isempty(grid::SparseVoxelGrid) = length(grid) == 0
function Base.show(io::IO, grid::SparseVoxelGrid)
    println(io, typeof(grid))
    println(io, "  Number of voxels: ", length(grid.ind_range))
    println(io, "  Number of points in grid: ", length(grid.indices))
    print(io, "  Voxel side length: ", grid.voxel_size)
end

"""
    invoxel(grid::SparseVoxelGrid, voxel_id::Vector)

Returns an iterator to retrieve point indices in the `voxel_id`.
"""
function invoxel(grid::SparseVoxelGrid, voxel_id::VoxelId)
    ind_range = get(grid.ind_range, voxel_id, 0:-1)
    VoxelIndicesIter(grid.indices, ind_range)
end

immutable VoxelIndicesIter
    indices::Vector{Int}
    range::UnitRange{Int}
end
# Functionality for VoxelIndicesIter
Base.start(indices::VoxelIndicesIter) = 1
Base.next(indices::VoxelIndicesIter, state) = indices.indices[indices.range[state]], state + 1
Base.done(indices::VoxelIndicesIter, state) = state > length(indices.range)
Base.eltype(::VoxelIndicesIter) = Int
function Base.show(io::IO, indices::VoxelIndicesIter)
    print(io, typeof(indices), " with ", length(indices.range), " voxel indices.")
end

"""
    voxelids(grid::SparseVoxelGrid) -> voxel_ids

Returns an iteratable object for the voxel ids in the grid.
"""
voxelids(grid::SparseVoxelGrid) = VoxelIDIter(grid.ind_range)

immutable VoxelIDIter
    dict::Dict{VoxelId, UnitRange{Int}}
end

# Functionality for VoxelIDIter
Base.length(v::VoxelIDIter) = length(v.dict)
Base.isempty(v::VoxelIDIter) = isempty(v.dict)
Base.start(grid::VoxelIDIter) = start(grid.dict)
function Base.next(grid::VoxelIDIter, state)
    n = next(grid.dict, state)
    n[1][1], n[2]
end
Base.done(grid::VoxelIDIter, state) = done(grid.dict, state)
Base.eltype(::VoxelIDIter) = VoxelId
function Base.show(io::IO, iter::VoxelIDIter)
    print(io, typeof(iter), " with ", length(iter.dict), " voxel ids.")
end
