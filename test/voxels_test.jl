@testset "Voxels" begin
    # Create 3x3 grid
    x_space = collect(linspace(0.0,3.0,3))
    y_space = x_space
    z_space = x_space
    x = [i for i in y_space, j in x_space, z in z_space]
    y = [j for i in y_space, j in x_space, z in z_space]
    z = [z for i in y_space, j in y_space, z in z_space]
    points = (hcat([[x[i], y[i], z[i]] for i = 1:length(x)]...))
    voxels = make_voxels(points, 1.0)

    # Check that each point is in one voxel
    for i = 1:length(voxels)
        @test length(voxels[i].point_inds) == 1
        @test voxels[i].point_inds[1] == i
    end

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
    voxels = make_voxels(points, 1.0)
    @test voxels[1].point_inds == [1,5,6]
    @test voxels[2].point_inds == [2]
    @test voxels[3].point_inds == [3]
    @test voxels[4].point_inds == [4]
    @test voxels[5].point_inds == [7]
    @test_approx_eq_eps voxels[1].position [1.0,1.0,0.0] 1e-2
    @test_approx_eq_eps voxels[2].position [2.0,1.0,0.0] 1e-2
    @test_approx_eq_eps voxels[3].position [1.0,2.0,0.0] 1e-2
    @test_approx_eq_eps voxels[4].position [2.0,2.0,0.0] 1e-2
    @test_approx_eq_eps voxels[5].position [10.0^5,10.0^5,0.0] 1e-2

    # Test using point clouds
    cloud = PointCloud([Vec(i+1.0, i-1.0, i) for i = 1:10])
    voxels = make_voxels(cloud, 1.0)
    for i = 1:length(voxels)
        @test length(voxels[i].point_inds) == 1
        @test voxels[i].point_inds[1] == i
    end
end
