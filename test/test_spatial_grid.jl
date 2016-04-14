@testset "Voxels" begin
    @testset "3x3 Voxel Set" begin
        # Create 3x3 grid
        x_space = collect(linspace(0.0,3.0,4))
        y_space = x_space
        z_space = x_space
        x = [i for i in y_space, j in x_space, z in z_space]
        y = [j for i in y_space, j in x_space, z in z_space]
        z = [z for i in y_space, j in y_space, z in z_space]
        points = (hcat([[x[i], y[i], z[i]] for i = 1:length(x)]...))
        voxel_size = 2.0
        voxels = Voxel(points, voxel_size)
        inds = invoxel(voxels)
        @test voxels.voxel_size == voxel_size
        @test voxels.offset == zeros(3)
        @test voxels.n_voxels == [2.0,2.0,2.0]
    end

    @testset "2x2 Flat Grid" begin
        # Create a 2x2 flat grid
        xy_space = collect(linspace(0.0,1.0,2))
        x = [i for i in xy_space, j in xy_space]
        y = [j for i in xy_space, j in xy_space]
        points = hcat([[x[i], y[i], 0.0] for i = 1:length(x)]...)

        # Test border points and assignments
        p = [0.99 0.0 10.0^5
             0.0  0.99 10.0^5
             0.0  0.0 0.0]

        points = [points p]
        voxels = Voxel(points, 1.0)
        @test invoxel(voxels, (1,1,1)) == [1,5,6]
        @test invoxel(voxels, (2,1,1)) == [2]
        @test invoxel(voxels, (1,2,1)) == [3]
        @test length(invoxel(voxels)) == 5

        centres = [0.5  0.5  1.5  1.5  1.000005e5
                   0.5  1.5  0.5  1.5  1.000005e5
                   0.5  0.5  0.5  0.5  0.5]
        [@test_approx_eq_eps voxels.centres[:,i] centres[:,i] 1e-5 for i = 1:size(centres,2)]
    end

    @testset "Voxelise cloud" begin
        cloud = PointCloud([Vec(mod(i+1.0,2), i-1.0, i) for i = 1:10])
        voxels = Voxel(cloud, 2.0)
        inds = invoxel(voxels)
        @test inds[1] == [1,2]
        @test inds[2] == [3,4]
        @test inds[3] == [5,6]
        @test inds[4] == [7,8]
        @test inds[5] == [9,10]
    end
end
