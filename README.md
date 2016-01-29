# PointClouds

A package for flexible point cloud data handling

The aim here is to have a point cloud data structure with spatial lookup and
arbitrary per-point attributes, along with some basic IO and utility
functions.  From an attribute access and manipulation point of view, a
`PointCloud` is very much like a `DataFrame`.

Perhaps one day `PointCloud` can be implemented in terms of an underlying
`DataFrame`, but at this stage the `DataFrames` package has a naturally strong
focus on statistical computation, which seems somewhat at odds with the
geometrically local spatial computation which one often wants with a point
cloud.

Hopefully this package will soon make sense as an open source component (hence
the MIT license), but we should develop a core solid design first so we know
exactly what it should do.
