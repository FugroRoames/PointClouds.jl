using PointClouds
reload("PointClouds")
using StaticArrays

points_SA = [SVector{3,Float64}(rand(3)) for i = 1:10]
pc1 = PointCloud(points_SA)

points = rand(3,10)
pc2 = PointCloud(points)
inrange(pc2, points, 10.0)
knn(pc2, points, 2)

pos = [1.0,2.0,3.0]
inrange(pc1, pos, 10.0)
knn(pc1, pos, 2)

pos_SV = SVector{3,Float64}(1.0, 2.0, 3.0)
inrange(pc1, pos_SV, 10.0)
knn(pc1, pos_SV, 2)

points_3 = [SVector{3,Float64}(rand(3)) for i = 1:5]
pc3 = PointCloud(points_3)

out = vcat(pc1, pc2, pc3)
