
type Voxel
    id         :: Int64
    position   :: Vector{Float64}
    point_inds :: Vector{Int64}
end

function make_voxels(points::PointCloud, voxel_size)
    make_voxels(destructure(points.positions), voxel_size)
end

function make_voxels{T<:AbstractFloat}(points::Array{T,2}, voxel_size::Real)
    ndims, npoints = size(points)
    ndims > 1 || error("Dimensions of points should be greater than a 1xN")
    buffer = 1e-3

    bounds = Array(Float64, 2, ndims)
    n = Array(Float64, ndims)
    d = Array(Float64, ndims)
    for i = 1:ndims
        min, max = extrema(points[i,:])
        bounds[1, i] = min - buffer
        bounds[2, i] = max + buffer
        n[i] = ceil((bounds[2, i] - bounds[1, i]) / voxel_size)
        d[i] = (bounds[2, i] - bounds[1, i]) / n[i]
    end

    #update the voxel indices
    point_voxel_inds = zeros(Int64, npoints)
    c = zeros(Float64, ndims)
    for i = 1:npoints
        c[1] = ceil((points[1, i] - bounds[1,1]) / d[1])
        ind = c[1]
        for j = 2:ndims
            c[j] = ceil((points[j, i] - bounds[1,j]) / d[j])
            ind += (c[j] - 1.0)*prod(n[1:j-1])
        end
        point_voxel_inds[i] = ind
    end

    active_voxels = unique(point_voxel_inds)
    nvoxels = length(active_voxels)
    voxels = Array(Voxel, nvoxels)

    # calculate voxel position, id and point inds
    voxel_pos = Array(Float64, ndims)
    level = Array(Float64, ndims)
    rem = Array(Float64, ndims)
    for j = 1:nvoxels
        voxel_ind = 0.0
        i = ndims
        level[i] =  ceil(active_voxels[j] ./ prod(n[1:i-1])) # Z level
        rem[i] = level[i] > 0.0  ? active_voxels[j] - (level[i] - 1.0) * prod(n[1:i-1]) : active_voxels[j]
        voxel_pos[i] = level[i] * d[i]
        c[i] = ceil((voxel_pos[i] - d[i] / 2.0) / d[i])
        voxel_ind = (c[i] - 1.0) * prod(n[1:i-1])
        for i = ndims-1:-1:2
            level[i] =  ceil(rem[i+1] ./ prod(n[1:i-1])) # Z level
            if level[i] > 0.0
                rem[i] = rem[i+1] - (level[i] - 1.0) * prod(n[1:i-1])
            else
                rem[i] = rem[i+1]
            end
            voxel_pos[i] = level[i] * d[i]
            c[i] = ceil((voxel_pos[i] - d[i] / 2.0) / d[i])
            voxel_ind += (c[i] - 1.0) * prod(n[1:i-1])
        end
        voxel_pos[1] = rem[2]*d[1]
        c[1] = ceil((voxel_pos[1] - d[1] / 2.0) / d[1])
        voxel_ind += c[1]
        voxels[j] = Voxel(Int64(voxel_ind), copy(voxel_pos), Int64[])
    end

    #TODO: It would probably be better to make a type "Voxels" with an array of voxels and this dict attached
    indlookup = Dict([(voxels[i].id,i) for i = 1:length(voxels)])

    for i = 1:npoints
        voxel_id = point_voxel_inds[i]
        voxel_ind = indlookup[voxel_id]
        push!(voxels[voxel_ind].point_inds,i)
    end
    return voxels
end


function make_voxels_old(points::Array{Float64,2}, voxel_size::Real)
    npoints = size(points, 2)
    point_voxel_inds = zeros(Int64, npoints)
    buffer = 1e-3

    xmin = minimum(points[1, :]) - buffer
    xmax = maximum(points[1, :]) + buffer
    ymin = minimum(points[2, :]) - buffer
    ymax = maximum(points[2, :]) + buffer
    zmin = minimum(points[3, :]) - buffer
    zmax = maximum(points[3, :]) + buffer
    nx = ceil((xmax - xmin) / voxel_size)
    ny = ceil((ymax - ymin) / voxel_size)
    nz = ceil((zmax - zmin) / voxel_size)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    #update the voxel indices
    for i = 1:npoints
        cx = ceil((points[1, i] - xmin) / dx)
        cy = ceil((points[2, i] - ymin) / dy)
        cz = ceil((points[3, i] - zmin) / dz)
        point_voxel_inds[i] = ((cz - 1.0) * ny * nx + (cy - 1.0) * nx + cx)
    end

    active_voxels = unique(point_voxel_inds)
    nvoxels = length(active_voxels)
    voxels = Array(Voxel, nvoxels)

    #calculate voxel position and id and point inds
    for j = 1:nvoxels
        zlevel = ceil(active_voxels[j] ./ (nx * ny))
        if zlevel > 0.0
            zrem = active_voxels[j] - (zlevel - 1.0) * nx * ny
        else
            zrem = active_voxels[j]
        end
        ylevel = ceil(zrem ./ nx)
        if ylevel > 0.0
            ymod = zrem - (ylevel - 1.0) * nx
        else
            ymod = zrem
        end
        voxel_pos = [ymod * dx, ylevel * dy, zlevel * dz]
        cx = ceil((voxel_pos[1] - dx / 2.0) / dx)
        cy = ceil((voxel_pos[2] - dy / 2.0) / dy)
        cz = ceil((voxel_pos[3] - dz / 2.0) / dz)
        voxel_ind = ((cz - 1.0) * ny * nx + (cy - 1.0) * nx + cx)
        voxels[j] = Voxel(voxel_ind,voxel_pos, Int64[])
    end

    #TODO: It would probably be better to make a type "Voxels" with an array of voxels and this dict attached
    indlookup = Dict([(voxels[i].id,i) for i = 1:length(voxels)])

    for i = 1:npoints
        voxel_id = point_voxel_inds[i]
        voxel_ind = indlookup[voxel_id]
        push!(voxels[voxel_ind].point_inds,i)
    end
    return voxels
end
