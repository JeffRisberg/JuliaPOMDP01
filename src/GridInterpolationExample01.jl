#=
GridInterpolationExample01:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-14
=#
using GridInterpolations

grid = RectangleGrid([0., 0.5, 1.],[0., 0.5, 1.])  	# rectangular grid
sGrid = SimplexGrid([0., 0.5, 1.],[0., 0.5, 1.])	# simplex grid
gridData = [8., 1., 6., 3., 5., 7., 4., 9., 2.]   	# vector of value data at each cut
point = [0.25, 0.75]  								# point at which to perform interpolation

println(interpolate(grid, gridData, point))

println(interpolate(sGrid, gridData, point))

println(interpolants(grid, point))

println(interpolants(sGrid, point))

grid = RectangleGrid([0., 0.5, 1.],[0., 0.5, 1.])  	# rectangular grid
println(interpolants(grid, point))
# the interpolants are a list of point indexes which are to be used, and list of weights for those points.
# in a simplex grid there will be three of each, and in a rectangle grid there will be four of each.

grid = RectangleGrid([0., 5.0, 10.0],[0., 0.5, 1.])  	# rectangular grid
println(interpolants(grid, [5.0, 0.5]))
