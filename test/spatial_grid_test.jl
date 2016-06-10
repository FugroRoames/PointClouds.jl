@testset "SparseVoxelGrid tests" begin
    @testset "Voxelization" begin
        # Create points for a uniformly spaced 4x4x4 grid
        x = collect(linspace(0.0, 3.0, 4))
        cnt = 1
        points = zeros(3, 64)
        for i in x, j in x, k in x
            points[1, cnt] = i
            points[2, cnt] = j
            points[3, cnt] = k
            cnt += 1
        end

        # Test standard uniformly sized voxels
        voxel_size = 1.51
        grid = SparseVoxelGrid(points, voxel_size)
        @test length(collect(grid)) == 8
        [@test length(collect(voxel)) == 8 for voxel in grid]

        # Test a point cloud
        cloud = PointCloud(points)
        grid = SparseVoxelGrid(cloud, voxel_size)
        [@test length(collect(voxel)) == 8 for voxel in grid]

        # Test voxels with different side lengths in each axis
        grid = SparseVoxelGrid(cloud, voxel_size, voxel_size, 4.0)
        [@test length(collect(voxel)) == 16 for voxel in grid]
        grid = SparseVoxelGrid(points, voxel_size, 4.0, 4.0)
        [@test length(collect(voxel)) == 32 for voxel in grid]
    end

    @testset "Neighbouring voxel" begin
        srand(1)
        points = rand(3, 10000) * 10.0
        grid = SparseVoxelGrid(points, 5.0)
        radius = 1
        for voxel in grid
            @test length(collect(in_cuboid(grid, voxel, radius))) == 7
        end

        grid = SparseVoxelGrid(points, 2.5)
        voxel_list = [(1,1,1), (1,2,1), (2,2,2)]
        @test length(collect(in_cuboid(grid, voxel_list[1], radius))) == 26
        @test length(collect(in_cuboid(grid, voxel_list[2], radius))) == 26
        @test length(collect(in_cuboid(grid, voxel_list[3], 2))) == length(grid)-1

        in_cuboid(grid, (1,1,1), radius) do voxel
            @test haskey(grid, voxel.id)
        end
    end
end
