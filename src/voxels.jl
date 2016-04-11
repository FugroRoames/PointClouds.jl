type Voxel{T,N}
    centres::Matrix{T}            # Centre points of voxels
    indices::Vector{NTuple{N,T}}  # Voxel indices
    voxel_size::Real              # Size of each voxel
    n_voxels::Vector{Float64}     # TODO This may not be that useful?
    offset::Vector{T}             # TODO Should this be centre?
    ind_range::Dict{NTuple{N, T}, UnitRange{Int64}}  # Point position range for voxel index
    ind_lookup::Dict{Int64,Int64}  # Dictionary lookup
end

function Voxels{T <: AbstractFloat}(points::Matrix{T},
                                    voxel_size::Real;
                                    offset::Bool = false)
    ndims, npoints = size(points)
    centred_points = minimum(points, 2)
    points = points .- centred_points

    # Count number voxels in each dimension
    n_voxels = Array(Float64, ndims)
    for i = 1:ndims
        n_voxels[i] = ceil(maximum(points[i, :]) / voxel_size)
    end

    # Get indices for voxels
    indices = get_indices(points, voxel_size)

    # Sort point indices to get easy range look up for each voxel
    permutation = sortperm(indices)
    sorted_ind = indices[permutation]
    unique_ind = unique(sorted_ind)

    # Create a dictionary assigning voxel indices to UnitRange of points
    indices_range = get_indices_range(unique_ind, sorted_ind)

    # Look up to map the UnitRange to original point indices
    ind_lookup = Dict{Int64,Int64}()
    for i in eachindex(permutation)
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

function get_centres(indices, voxel_size)
    ndims = length(indices[1])
    nvoxels = length(indices)
    # indices position (middle point of indices)
    pos = Array(eltype(eltype(indices)), ndims, nvoxels)
    for i in 1:nvoxels
        pos[:,i] = collect(indices[i])*voxel_size - voxel_size*0.5
    end
    return pos
end

function get_centre{N, T}(voxels::Voxel, voxel_index::NTuple{N,T}; offset::Bool = false)
    position = collect(voxel_index)*voxels.voxel_size - voxels.voxel_size*0.5
    position = offset ? position : position .+ voxels.offset
end

"""
Calculates voxel indices from points

points 3xN array
number of voxels in each dimension N vector
voxel_size
"""
function get_indices{T <: AbstractFloat}(points::Matrix{T}, voxel_size)
    ndims, npoints = size(points)
    indices = Array(NTuple{ndims,T}, npoints)
    for i in 1:npoints
        indices[i] = tuple(floor(points[:,i]./voxel_size) + 1.0 ...)
    end
    return indices
end

# Get the point range for each of the same voxel indices
# Assign the point ranges to voxel indices
function get_indices_range(unique_ind, sorted_ind)
    # zero crossing (i.e. when the indices change)
    zc = sum(abs(diff(hcat([collect(sorted_ind[i]) for i=1:length(sorted_ind)]...),2)),1)
    zc_ind = find(zc .!= 0.0)
    push!(zc_ind, length(sorted_ind))  # Add last point

    start_ind = 1
    indices_range = Dict{eltype(unique_ind), UnitRange{Int64}}()
    for (i, ind) in enumerate(zc_ind)
        indices_range[unique_ind[i]] = start_ind:ind
        start_ind = ind + 1
    end
    return indices_range
end

# Return point indices for given query voxel indices
function invoxel{N, T}(voxels::Voxel, voxel_indices::Vector{NTuple{N,T}})
    num_query_voxels = length(voxel_indices)
    indices = Array(Vector{Int64}, num_query_voxels)
    for i = 1:num_query_voxels
        indices[i] = invoxel(voxels, voxel_indices[i])
    end
    return indices
end

# Return point indices for a voxel index
function invoxel{N, T}(voxels::Voxel, voxel_index::NTuple{N,T})
    range = get(voxels.ind_range, voxel_index, 0:0)
    if range != 0:0
        indices = Int64[voxels.ind_lookup[j] for j in range]
    else
        indices = Int64[]
    end
    return indices
end

# Return point indices for all voxels
invoxel(voxels::Voxel) = invoxel(voxels, voxels.indices)


# Return voxel index given a position
function invoxel{T <: AbstractFloat}(voxels::Voxel, point::Vector{T})
    point = point .- voxels.offset # Centre into voxelized frame
    return tuple(floor(point./voxels.voxel_size) + 1.0 ...)
end

# Voxelize point cloud
Voxels(cloud::PointCloud, voxel_size) = Voxels(destructure(cloud.positions), voxel_size)

function Base.show(io::IO, voxels::Voxel)
    println(io, typeof(voxels))
    println(io, "  Number of dimensions: ", size(voxels.centres, 1))
    println(io, "  Number of voxels in each dimension: ", voxels.n_voxels)
    println(io, "  Size of voxels: ", voxels.voxel_size)
    print(io, "  Offset: ", voxels.offset)
end
