@testset "Voxels" begin
    @testset "3x3 point grid voxelization " begin
        # Create a 3x3 grid
        x_space = collect(linspace(0.0, 3.0, 4))
        y_space = x_space
        z_space = x_space
        x = [i for i in y_space, j in x_space, z in z_space]
        y = [j for i in y_space, j in x_space, z in z_space]
        z = [z for i in y_space, j in y_space, z in z_space]
        points = (hcat([[x[i], y[i], z[i]] for i = 1:length(x)]...))
        voxel_size = 1.5
        grid = voxelize(points, voxel_size)
        for voxel_id in voxelids(grid)
            indices = invoxel(grid, voxel_id)
            @test length(collect(indices)) == 8
        end
    end

    @testset "Voxelize point cloud" begin
        # points in a straight line
        cloud = PointCloud([Vec(i, 0.0, 0.0) for i = 1:10])
        grid = voxelize(cloud, 2.0)
        for voxel_id in voxelids(grid)
            indices = invoxel(grid, voxel_id)
            @test length(collect(indices)) == 2
        end
    end
end
