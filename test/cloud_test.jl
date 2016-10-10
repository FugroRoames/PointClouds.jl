
@testset "PointClouds tests" begin

typealias V3d SVector{3,Float64}

# Unit cube of points in 3D
position = [0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  1.0  1.0  1.0  0.0
            0.0  0.0  0.0  1.0  1.0  0.0  1.0  1.0
           ]
cloud = PointCloud(position)
cloud[:intensity] = collect(1:8)

@testset "Column access" begin
    @test haskey(cloud, :intensity)
    @test sort(collect(keys(cloud))) == [:intensity]

    @test cloud[:intensity][1]   == 1
    @test cloud[:intensity][end] == 8

    @test positions(cloud) === cloud.positions
end

@testset "Row access" begin
    @test length(cloud) == 8

    subset = cloud[3:4]
    @test positions(subset)[1] == V3d(1,1,0)
    @test subset[:intensity][2] == 4

    dupcloud = [cloud; cloud]
    @test positions(dupcloud[1:8]) == positions(dupcloud[9:end])
    @test dupcloud[1:8][:intensity] == dupcloud[9:end][:intensity]

    splitres = split_cloud(cloud, cloud[:intensity] .% 2)
    @test length(splitres[1]) == 4
    @test splitres[1][:intensity] == collect(1:2:8)
    @test length(splitres[0]) == 4
    @test splitres[0][:intensity] == collect(2:2:8)
end

@testset "Spatial indexing" begin
    @test knn(cloud, [0,0,0], 1)[1] == [1]
    @test sort(knn(cloud, V3d(0,0,0), 4)[1]) == [1,2,6,8]
    @test inrange(cloud, [0,0,0], 0.1) == [1]
    @test sort(inrange(cloud, V3d(0,0,0), 1.1)) == [1,2,6,8]

    # multiple points
    multi = [V3d(0,0,0), V3d(1,1,1)]
    @test knn(cloud, multi, 1)[1] == [[1], [5]]
    @test inrange(cloud, multi, 1.1) == [[1,2,6,8], [3,4,5,7]]
end

@testset "PCA" begin
    centres = [
        V3d(0,0,0),
        V3d(10,0,0),
        V3d(0,10,0),
    ]
    desired_normals = [
        V3d(1,0,0),
        V3d(1,1,0)/sqrt(2),
        V3d(1,0,1)/sqrt(2),
    ]

    position = V3d[]
    for j=1:3
        Nj = desired_normals[j]
        for i=1:10
            r = V3d(randn(),randn(),randn())
            r = r - dot(Nj, r)*Nj
            push!(position, r + centres[j])
        end
    end

    cloud = PointCloud(position)

    add_normals!(cloud)

    N = cloud[:normals]
    N1 = N[knn(cloud, centres[1], 1)[1][1]]
    N2 = N[knn(cloud, centres[2], 1)[1][1]]
    N3 = N[knn(cloud, centres[3], 1)[1][1]]
    @test abs(abs(dot(V3d(N1), desired_normals[1])) - 1) < 10*eps(Float32)
    @test abs(abs(dot(V3d(N2), desired_normals[2])) - 1) < 10*eps(Float32)
    @test abs(abs(dot(V3d(N3), desired_normals[3])) - 1) < 10*eps(Float32)
end

end # testset
