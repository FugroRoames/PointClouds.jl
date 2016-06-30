@testset "SparseVoxelGrid tests" begin
    # Create points for a uniformly spaced 4x4x4 grid
    x = collect(linspace(0.0, 3.0, 3))
    cnt = 1
    points = zeros(3, 27)
    for i in x, j in x, k in x
        points[1, cnt] = i
        points[2, cnt] = j
        points[3, cnt] = k
        cnt += 1
    end
    cloud = PointCloud(points)

    @testset "Rasterizer" begin
        # Test standard uniformly sized voxels
        d = rasterize_points(points, 1.0)
        [@test length(r) == 3 for r in values(d)]

        # Test a point cloud
        d = rasterize_points(cloud, 1.0)
        [@test length(r) == 3 for r in values(d)]
    end

    @testset "Voxelization" begin
        # Test standard uniformly sized voxels
        voxel_size = 1.0
        grid = SparseVoxelGrid(points, voxel_size)
        @test isempty(grid) == false
        @test haskey(grid, (0,0,0)) == true
        @test length(grid) == 27
        @test length(collect(grid)) == 27
        [@test length(collect(voxel)) == 1 for voxel in grid]

        grid = SparseVoxelGrid(cloud, voxel_size)
        [@test length(collect(voxel)) == 1 for voxel in grid]
        @test voxel_center(grid, (1, 1, 1)) == Vec(0.5, 0.5, 0.5)

        # Test voxels with different side lengths in each axis
        @test length(collect(SparseVoxelGrid(cloud, (2.0, 2.5, 4.0)))) == 4
    end

    @testset "Neighbouring voxel" begin
        radius = 1
        grid = SparseVoxelGrid(points, 2.0)
        for voxel in grid
            # Test using anonymous function method
            in_cuboid(grid, voxel, radius) do voxel
                @test haskey(grid, voxel.id)
            end
        end

        grid = SparseVoxelGrid(points, 1.5)
        voxel_list = [(1,1,1), (1,2,1), (2,2,2), (10,10,10)]
        @test length(collect(in_cuboid(grid, voxel_list[1], 2))) == 26
        @test length(collect(in_cuboid(grid, voxel_list[2], 2))) == 26
        @test length(collect(in_cuboid(grid, voxel_list[3], 2))) == 26
        @test length(collect(in_cuboid(grid, voxel_list[4], 2))) == 0
        @test length(collect(in_cuboid(grid, collect(grid)[1], 2))) == 26

        # Test show methods
        io = IOBuffer()
        voxel = collect(grid)[1]
        cuboid = in_cuboid(grid, (0,0,0), radius)
        show(io, grid)
        show(io, voxel)
        show(io, cuboid)
    end
end
