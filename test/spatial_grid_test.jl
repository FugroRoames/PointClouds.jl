@testset "Voxels" begin
    @testset "3x3 point grid voxelization " begin
        # Create a 3x3 grid
        x = collect(linspace(0.0, 3.0, 4))
        cnt = 1;
        points = zeros(3,64);
        for i in x, j in x, k in x
            points[1,cnt]  = i
            points[2,cnt] = j
            points[3,cnt] = k
            cnt +=1
        end
        voxel_size = 1.5
        grid = SparseVoxelGrid(points, voxel_size)
        @test length(collect(grid)) == 8
        for voxel in grid
            @test length(collect(voxel)) == 8
        end

        cloud = PointCloud(points)
        grid = SparseVoxelGrid(cloud, voxel_size)
        for voxel in grid
            @test length(collect(voxel)) == 8
        end
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
        @test length(collect(in_cuboid(grid, voxel_list[1], radius))) == 7
        @test length(collect(in_cuboid(grid, voxel_list[2], radius))) == 11
        @test length(collect(in_cuboid(grid, voxel_list[3], 2))) == length(grid)-1

        in_cuboid(grid, (1,1,1), radius) do voxel
            @test haskey(grid, voxel.id)
        end
    end
end
