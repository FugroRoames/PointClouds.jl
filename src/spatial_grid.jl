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
    SparseVoxelGrid(points, voxel_size) -> grid

Creates a sparse spatial grid by organising 3D points into voxels. `points` can either
be a 3xN matrix or a `PointCloud`. The `voxel_size` is the side length of each cell.

### Example

To create a spatial grid with voxel side length of 10 metres for arbitrary points:
```julia
using PointClouds
points = rand(3, 100000) * 20.0
grid = SparseVoxelGrid(points, 10.0)
```

The created grid is an iteratable object which returns a `Voxel` in each iteration.
Each voxel can be accessed directly with a `for` loop or all voxels can be `collect`ed into an array.
Likewise, the returned `Voxel` is an iterable object that returns the point indices:
```julia
# Iterate through each voxel in grid
for voxel in grid
    # Get each point index in voxel
    for idx in voxel
        # Do stuff with points[:,idx]
    end
    # Or, you may want all point indices in a voxel
    all_point_indices = collect(voxel)
end
```
"""
immutable SparseVoxelGrid{T <: AbstractFloat}
    voxel_size::T
    voxel_info::Dict{VoxelId, UnitRange{Int}}
    point_indices::Vector{Int}
end

function SparseVoxelGrid{T <: AbstractFloat}(points::Matrix{T}, voxel_size::T)
    ndims, npoints = size(points)
    ndims != 3 && throw(ArgumentError("Points dimensions are $(size(points)), should be a 3xN matrix."))

    # In order to avoid allocating a vector for each voxel, we construct the data structure in a backward-looking order.

    # Assign each point to a voxel id
    voxel_ids = Vector{VoxelId}(npoints)
    for j = 1:npoints
        voxel_ids[j] = make_voxel_id(points, j, voxel_size)
    end

    # Count the number of points in each voxel
    group_counts = Dict{VoxelId, Int}()
    for id in voxel_ids
        group_counts[id] = get(group_counts, id, 0) + 1
    end

    # Allocate ranges for the indices of points in each voxel based on the counts
    voxel_info = Dict{VoxelId, UnitRange{Int64}}()
    current_index = 1
    for (group_id, group_size) in group_counts
        voxel_info[group_id] = current_index:current_index+group_size-1
        current_index += group_size
    end

    # Place indices for points into the appropriate index range for the associated voxel
    point_indices = Vector{Int}(npoints)
    for j = 1:npoints
        id = voxel_ids[j]
        index_in_group = group_counts[id]
        group_counts[id] = index_in_group - 1
        point_indices[first(voxel_info[id]) + index_in_group-1] = j
    end

    return SparseVoxelGrid(voxel_size, voxel_info, point_indices)
end

# Voxelize point cloud
SparseVoxelGrid(cloud::PointCloud, voxel_size::AbstractFloat) = SparseVoxelGrid(destructure(cloud.positions), voxel_size)

Base.length(grid::SparseVoxelGrid) = length(grid.voxel_info)
Base.isempty(grid::SparseVoxelGrid) = isempty(grid.voxel_info)
Base.haskey(grid::SparseVoxelGrid, k) = haskey(grid.voxel_info, k)
function Base.show(io::IO, grid::SparseVoxelGrid)
    println(io, typeof(grid))
    println(io, "  Number of voxels: ", length(grid))
    println(io, "  Number of points in grid: ", length(grid.point_indices))
    print(io, "  Voxel side length: ", grid.voxel_size)
end

"""
    make_voxel_id(points::Matrix, index, voxel_size)

Create a 3D voxel id tuple for the point specified by the column index.
"""
function make_voxel_id(points::Matrix, index, voxel_size::AbstractFloat)
    (floor(Int, points[1, index] / voxel_size), floor(Int, points[2, index] / voxel_size),
     floor(Int, points[3, index] / voxel_size))
end

"An iterator type to return point indices in a voxel. See SparseVoxelGrid() for usage."
immutable Voxel
    id::VoxelId
    point_index_range::UnitRange{Int}
    all_point_indices::Vector{Int}
end

Base.start(grid::SparseVoxelGrid) = start(grid.voxel_info)
function Base.next(v::SparseVoxelGrid, state)
    n = next(v.voxel_info, state)
    id = n[1][1]
    point_index_range = n[1][2]
    Voxel(id, point_index_range, v.point_indices), n[2]
end
Base.done(grid::SparseVoxelGrid, state) = done(grid.voxel_info, state)
Base.eltype(::SparseVoxelGrid) = Voxel
function Base.getindex(grid::SparseVoxelGrid, id::VoxelId)
    Voxel(id, grid.voxel_info[id], grid.point_indices)
end

Base.start(v::Voxel) = 1
function Base.next(v::Voxel, state)
    v.all_point_indices[v.point_index_range[state]], state + 1
end
Base.done(v::Voxel, state) = state > length(v.point_index_range)
Base.eltype(::Voxel) = Int
Base.length(v::Voxel) = length(v.point_index_range)
function Base.show(io::IO, v::Voxel)
    print(io, typeof(v), " ", v.id, " with ", length(v.point_index_range), " points")
end

"Voxel iterator that returns the `Voxel`s. See `in_cuboid()` for usage."
immutable VoxelCuboid
    grid::SparseVoxelGrid
    voxel_id::VoxelId
    range::CartesianRange{CartesianIndex{3}}
end

# TODO the `do` syntax for in_cuboid is faster than the iterator - can the iterator be improved?

"""
    in_cuboid(grid::SparseVoxelGrid, voxel::Voxel, radius::Int)
    in_cuboid(grid::SparseVoxelGrid, voxel_id::NTuple{3,Int64}, radius::Int)

Search for neighbouring voxels within a `radius` around the reference `voxel` or `voxel_id`.
Returns a `Voxel` in each iteration.

### Example
The `in_cuboid` function can be implemented using the `do` block syntax:

```julia
radius = 1
query_voxel = (1,1,1)
in_cuboid(grid, query_voxel, radius) do voxel
    for index in voxel
        # Do stuff with point[:, index]
    end
    # Or, collect all indices into an array
    indices = collect(voxel)
end
```

Alternatively, you may use a `for` loop which returns a voxel in each iteratation:
```julia
for voxel in in_cuboid(grid, query_voxel, radius)
    # do stuff with the `Voxel` (i.e. collect(voxel) or for index in voxel etc.)
end
```
"""
in_cuboid(grid::SparseVoxelGrid, voxel::Voxel, radius::Int) = in_cuboid(grid, voxel.id, radius)

function in_cuboid(grid::SparseVoxelGrid, voxel::VoxelId, radius::Int)
    start = CartesianIndex((-radius+voxel[1], -radius+voxel[2], -radius+voxel[3]))
    stop = CartesianIndex((radius+voxel[1], radius+voxel[2], radius+voxel[3]))
    VoxelCuboid(grid, voxel, CartesianRange(start, stop))
end

in_cuboid(f::Function, grid::SparseVoxelGrid, voxel::Voxel, radius::Int) = in_cuboid(f, grid, voxel.id, radius)

function in_cuboid(f::Function, grid::SparseVoxelGrid, voxel::VoxelId, radius::Int)
    for i=-radius+voxel[1]:radius+voxel[1], j=-radius+voxel[2]:radius+voxel[2], k=-radius+voxel[3]:radius+voxel[3]
        id = (i, j, k)
        if haskey(grid, id) && id != voxel
           f(grid[id])
        end
    end
end

function Base.getindex(c::VoxelCuboid, id::CartesianIndex{3})
    Voxel(id.I, c.grid.voxel_info[id.I], c.grid.point_indices)
end

Base.haskey(c::VoxelCuboid, next_id::CartesianIndex{3}) = haskey(c.grid, next_id.I)

function Base.start(c::VoxelCuboid)
    state = start(c.range)
    if !haskey(c.grid, state.I) # first voxel id is not in grid
        # find the next voxel in grid
        while !done(c.range, state)
            id, state = next(c.range, state)
            if haskey(c.grid, id.I) && c.voxel_id != id.I
                # return the voxel id
                return id, 1
            end
        end
        # no voxel id was found set value to quit iterations
        return state, 0
    end
    # return the starting voxel
    return state, 1
end
function Base.next(c::VoxelCuboid, state::Tuple{CartesianIndex{3}, Int64})
    next_state = state[1]
    voxel = c[next_state]
    # find the next voxel
    while !done(c.range, next_state)
        id, next_state = next(c.range, next_state)
        if haskey(c.grid.voxel_info, next_state.I) && c.voxel_id != next_state.I
            # return current voxel and the state for the next voxel
            return voxel, (next_state, 1)
        end
    end
    # Next voxel does not exist exists
    return voxel, (next_state, 0)
end
Base.done(c::VoxelCuboid, state) = state[2] == 0 || state[1][3] > c.range.stop[3]
Base.eltype(::VoxelCuboid) = Voxel
if VERSION >= v"0.5.0-dev+3305"
    # See https://github.com/JuliaLang/julia/issues/15977
    # Possibly could implement length() instead, but it's nontrivial work to compute.
    Base.iteratorsize(::Type{VoxelCuboid}) = Base.SizeUnknown()
end
function Base.show(io::IO, c::VoxelCuboid)
    print(io, typeof(c), " ID iteration range: ", c.range.start.I, " -> ", c.range.stop.I)
end

"""
    voxel_center(grid::SparseVoxelGrid, voxel_id::NTuple{3,Int64})

Calculate the centre point for the `voxel_id` in the spatial grid.
"""
function voxel_center(grid::SparseVoxelGrid, voxel_id::VoxelId)
    centre = collect(voxel_id) * grid.voxel_size - grid.voxel_size * 0.5
end
