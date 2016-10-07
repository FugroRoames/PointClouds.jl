using PointClouds
reload("PointClouds")
using StaticArrays

points = [SVector{3,Float64}(rand(3)) for i = 1:10]
pc = PointCloud(points)

pos = [1.0,2.0,3.0]
inrange(pc, pos, 10.0)
knn(pc, pos, 2)

pos = SVector{3,Float64}(1.0,2.0,3.0)
inrange(pc, pos, 10.0)
inrange(pc, points, 10.0)
knn(pc, pos, 2)



pc[1:5]

p2 = [SVector{3,Float64}(rand(3)) for i = 1:5]
pc2 = PointCloud(p2)
vcat(pc, pc2, pc2)
