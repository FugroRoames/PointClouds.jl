# PointClouds

A package for flexible point cloud data handling.

[![Build Status](https://travis-ci.org/FugroRoames/PointClouds.jl.svg?branch=master)](https://travis-ci.org/FugroRoames/PointClouds.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/bj9cu65wtb7a3j4t?svg=true)](https://ci.appveyor.com/project/c42f/pointclouds-jl)


## Basic usage


```julia
using FixedSizeArrays
using PointClouds

# Create positions of 3D Vec coordinates
positions = [Vec(i,-i,1.0) for i = 1:10]

# Create a PointCloud from positions
cloud = PointCloud(positions)

# Create new intensity attributes
cloud[:intensity] = collect(1:10)

# Create capture time for each point sample
cloud[:time] = [time() for i = 1:10]

# Find low intensity points
low_intensity_cloud = cloud[cloud[:intensity] .< 5]

# Find vector of time stamps per point
t = low_intensity_cloud[:time]

# Find points within 5 units of [1,1,1]
nearby = cloud[inrange(cloud, [1,1,1], 5.0)]
```


## Package goals

The aim here is to have a point cloud data structure with arbitrary per-point
attributes, spatial lookup, basic IO and various utility functions.

From an attribute access and manipulation point of view, a `PointCloud` is very
much like a `DataFrame`.  Perhaps one day `PointCloud` can be implemented in
terms of an underlying `DataFrame`, but at this stage the `DataFrames` package
has a naturally strong focus on statistical computation, which seems somewhat at
odds with the geometrically local spatial computation which one often wants with
a point cloud.

