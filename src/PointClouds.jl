module PointClouds

using NearestNeighbors

export PointCloud,
    # Point cloud data access
    positions, normals,
    # Data handling
    readdlm_points, split_cloud


"""
A PointCloud contains information about the 3D coordinates of the points in the
cloud, an efficient data structure for spatial search, and various attributes
which can be accessed by name.

    points - 3xN array containing the 3D coordinates of the points composing
             the cloud (TODO: Make this an array of fixed size arrays?)
"""
type PointCloud{T}
    points::Array{Float64, 2}
    tree::T
    attributes::Dict{Symbol,Array}
end


PointCloud(points) = PointCloud(points, KDTree(points), Dict{Symbol,Array}())

function Base.show(io::IO, pc::PointCloud)
    print(io, "PointCloud(N=$(length(pc)), attributes=$(keys(pc.attributes)))")
end

# Attribute access
Base.getindex(pc::PointCloud, i) = pc.points[:,i]

function Base.getindex(pc::PointCloud, attrname::Symbol)
    haskey(pc, attrname) || error("Point cloud $pc has no attribute $attrname")
    pc.attributes[attrname]
end

function Base.setindex!(pc::PointCloud, value::Array, attrname::Symbol)
    if size(value)[end] != length(pc)
        error("Length of attribute $attrname not equal to number of points = $(length(pc))")
    end
    pc.attributes[attrname] = value
end

Base.haskey(pc::PointCloud, attrname::Symbol) = (attrname == :position) || haskey(pc.attributes, attrname)

function Base.vcat{T}(cloud1::PointCloud{T}, clouds::PointCloud{T}...)
    # FIXME dispatch not seeing this...
    out = deepcopy(cloud1)
    ks = Set(keys(clouds1))
    for cloud in clouds
        ki = Set(keys(clouds))
        ki == ks || error("Cannot join clouds with inconsistent attributes")
        # FIXME: This is all an ugly hack!
        out.points = hcat(out.points, cloud.points)
        for (k,v) in out.attributes
            if ndims(val) == 1
                append!(v, cloud.attributes[k])
            else
                out.attributes[k] = hcat(out.attributes[k], cloud.attributes[k])
            end
        end
    end
    out
end

# Position attribute

"Return 3xN array of point cloud positions"
positions(pc::PointCloud)   = pc.points
Base.size(pc::PointCloud) = size(pc.points) # FIXME, this may not make sense?
Base.length(pc::PointCloud) = size(pc.points,2)

# Fast spatial knn lookup
NearestNeighbors.knn(pc::PointCloud, point, k) = knn(pc.tree, point, k)


# Column names and formats for Roames point cloud text format :(
const text_column_info = Dict(
    :X =>  (Float64, :position),  # TODO: Systematically aggregate position
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
    :L =>  (UInt16,  :point_source),
    :S =>  (Int64,   :flight_strip),
    :U =>  (Int,     :cluster),
    :Nx => (Float32, :normal_x), # TODO: Aggregate the normal info
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

    xyz_names = ["X", "Y", "Z"]
    xyz_cols = Int[findfirst(column_names, n) for n in xyz_names]
    # TODO: Use FixedSizeArrays for this?
    points = PointCloud(CloudId(-1,-1), data[:, xyz_cols]')

    for (cindex, cname) in enumerate(column_names)
        cname = Symbol(cname)
        cname ∉ xyz_names || continue
        ctype,fullname = text_column_info[cname]
        points[fullname] = map(ctype, data[:,cindex])
    end
    points
end

readdlm_points(filename::AbstractString) = open(io->readdlm_points(io), filename, "r")


"""
    pca_normals(points, tree; neighbours::Int=10)

Compute normals of a point cloud via PCA

For each point in the 3xN array `points`, compute an approximate surface normal
using PCA on the local neighbourhood, using the closest `neighbours` points.
`tree` should be an spatial lookup datastructure supporting the function `knn()`.
The normals are returned in a 3xN array.
"""
function pca_normals(points, tree; neighbours::Int=10)
    # Use Float32 for estimated normals - heaps of precision at half the memory
    normals = similar(points, Float32)
    for i = 1:size(points,2)
        p = points[:,i]
        inds,_ = knn(tree, p, neighbours)
        if isempty(inds)
            continue
        end
        d = (points[:,inds] .- p)
        # Select evec of smallest eval
        evals, evecs = eig(Symmetric(d*d'), 1:1)
        #print("$evals")
        normals[:,i] = evecs
    end
    return normals
end


"Return 3xN array of point cloud normals.  Generated on demand.

TODO: This is very specific to point matching at the moment.  For other uses,
we will want other ways to index the local environment, so generating this
magically on demand is really less than ideal.
"
function normals(cloud::PointCloud)
    if !haskey(cloud.attributes, :normal)
        cloud.attributes[:normal] = pca_normals(cloud.points, cloud.tree)
    end
    cloud.attributes[:normal]::Array{Float32,2}
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
        pos = positions(allpoints)[:,sel]
        cloud = PointCloud(CloudId(id,0), pos)
        for (name,val) in allpoints.attributes
            # FIXME: This is an ugly hack!
            if ndims(val) == 1
                cloud[name] = val[sel]
            else
                cloud[name] = val[:,sel]
            end
        end
        scans[id] = cloud
    end
    scans
end


end # module
