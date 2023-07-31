import open3d as o3d
import numpy as np
from math import tan as tan
import collections

'''
Goal of script: Put bounding boxes around each tree in original_point_cloud, and then use these
                to derive a carbon stock valuation for each one
'''

# CROP ORIGINAL IMAGE - MANUALLY GERNERATED BOUNDS IN ORDER TO ONLY INCLUDE USEFUL DATA
original_point_cloud = o3d.io.read_point_cloud("demo.ply")
#bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1., -1., -5.25), max_bound=(4.5, 4.5, -4.1))
#cropped_cloud = original_point_cloud.crop(bbox)
cropped_cloud = original_point_cloud

'''
TILING: The next section of the script uses tiling to break down our cropped cloud into subclouds
        so that we can apply our functions with more accuracy. This, of course, is at a cost of 
        running time - however we can alleviate this in many way.
'''

# STEP 1: Convert the point cloud to a numpy array
points = np.asarray(cropped_cloud.points)
colors = np.asarray(cropped_cloud.colors)

# STEP 2: Determine the bounding box
x_min, y_min, _ = np.min(points, axis=0)
x_max, y_max, _ = np.max(points, axis=0)

# STEP 3: Define the number of tiles in each dimension (4 chosen after some trial and error)
num_tiles_x = 4
num_tiles_y = 4

# STEP 4: Calculate the tile size
tile_size_x = (x_max - x_min) / num_tiles_x
tile_size_y = (y_max - y_min) / num_tiles_y

# STEP 5: Define a dict to hold the tiles, then assigned each point to a tile
tiles = collections.defaultdict(list)
tiles_colors = collections.defaultdict(list)

for idx, point in enumerate(points):
    tile_x = int((point[0] - x_min) // tile_size_x)
    tile_y = int((point[1] - y_min) // tile_size_y)
    
    # Due to rounding errors, a point may end up in an index equal to num_tiles_x/y.
    # In this case, we subtract 1 from the index.
    tile_x = min(tile_x, num_tiles_x - 1)
    tile_y = min(tile_y, num_tiles_y - 1)
    
    tiles[(tile_x, tile_y)].append(point)
    tiles_colors[(tile_x, tile_y)].append(colors[idx])

# STEP 6: Convert each tile to a point cloud
point_cloud_tiles = []
for (tile_x, tile_y), tile_points in tiles.items():
    tile_colors = tiles_colors[(tile_x, tile_y)]
    
    if tile_points:  # If the tile has points
        tile_point_cloud = o3d.geometry.PointCloud()
        tile_point_cloud.points = o3d.utility.Vector3dVector(tile_points)
        tile_point_cloud.colors = o3d.utility.Vector3dVector(tile_colors)
        point_cloud_tiles.append(tile_point_cloud)

'''
By this point in the script, we have now broken our cropped point cloud down into multiple smaller point clouds that
are strored in the variable: point_cloud_tiles.

Now, we define some functions that we will be applying to these tiles now that we have preprocessed our model
'''

# Function for finding local extremes - which is used to determine peaks of trees
# Gotten from https://stackoverflow.com/questions/27032562/local-maxima-in-a-point-cloud
from sklearn.neighbors import KDTree
def locally_extreme_points(coords, data, neighbourhood, lookfor = 'max', p_norm = 2.):
    '''
    Find local maxima of points in a pointcloud.  Ties result in both points passing through the filter.

    Not to be used for high-dimensional data.  It will be slow.

    coords: A shape (n_points, n_dims) array of point locations
    data: A shape (n_points, ) vector of point values
    neighbourhood: The (scalar) size of the neighbourhood in which to search.
    lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima
    p_norm: The p-norm to use for measuring distance (e.g. 1=Manhattan, 2=Euclidian)

    returns`
        filtered_coords: The coordinate`s of locally extreme points
        filtered_data: The values of these points
    '''
    assert coords.shape[0] == data.shape[0], 'You must have one coordinate per data point'
    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_radius(coords, r=neighbourhood)
    i_am_extreme = [data[i]==extreme_fcn(data[n]) for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme)  # This line just saves time on indexing
    return coords[extrema], data[extrema]

# Function for creating bounding boxes around each tree based on the local maxima generated in the variable extreme_points_pcd
# STEP 1: we put an arbritralily large bounding box around the max, such that we include the entire tree and then some.
# STEP 2: Tighten the bounding box by using the tree points that we found using the arbritrary box dimenstions
def create_bounding_boxes(extreme_points_pcd):
    # Set arbritrary dimenstions to the larger bounding box
    bounding_boxes=[]
    box_dimensions = [0.12, 0.12, 0.6]

    # Adding bounding box around max point and then some points around it
    for top_point in np.asarray(extreme_points_pcd.points):
        # Calculate the corner points of the bounding box
        min_bound = top_point - np.array(box_dimensions) / 2.0
        max_bound = top_point + np.array(box_dimensions) / 2.0

        # Extract points within the initial bounding box
        tree_points = original_point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
        tree_coords = np.asarray(tree_points.points)

        # Perform plane segmentation to separate the points belonging to the plane using ransac
        plane_model, inlier_indices = tree_points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
        non_plane_points = tree_points.select_by_index(inlier_indices, invert=True)
        non_plane_coords = np.asarray(non_plane_points.points)

        tree_as_pcd  = o3d.geometry.PointCloud()
        tree_as_pcd.points = o3d.utility.Vector3dVector(non_plane_coords)

        # Create tight bounding box around tree if we are able to put a bounding box around points
        if len(tree_as_pcd.points) > 3:
            try:
                bounding_box = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(tree_as_pcd)
                bounding_box.color = (1,0,0)
                bounding_boxes.append(bounding_box)
            except:
                continue
    return bounding_boxes

# Function for scaling sizes from model to real life - used to have our tree sizes have
def scale(drone_height, fov, full_pcd):
    theta = fov / 2
    x = tan(theta) * drone_height
    true_coverage = 2*x
    x_values = [point[0] for point in full_pcd.points]
    model_coverage = max(x_values) - min(x_values)
    return (true_coverage / model_coverage) * 10

# Function to get correct stock output (SUBJECT TO CHANGE - inaccurate for small trees)
def get_stock(height, width):
    # Using allometric equation
    return -3.190 + 1.145*np.log(width) + 2.092*np.log(height)


# Function for extracting features from a bouding box
def extract_features(box, num):
    minbound = box.get_min_bound()
    maxbound = box.get_max_bound()

    vertical_height = maxbound[2] - minbound[2]

    crown_width = max((maxbound[0] - minbound[0]),
                    (maxbound[1] - minbound[1]))

    scaled_height = vertical_height*scale(40,82,original_point_cloud)
    scaled_width = crown_width*scale(40,82,original_point_cloud)

    rounded_scaled_height = round(scaled_height, 2)
    rounded_scaled_width = round(scaled_width, 2)

    stock_level = get_stock(scaled_height, scaled_width)
    rounded_stock_level = round(stock_level, 3)

    return [num, rounded_scaled_height, rounded_scaled_width, rounded_stock_level]


the_bounding_boxes = []
# Now we are ready to apply our functions to each tile
for tile_cloud in point_cloud_tiles:
    if not np.asarray(tile_cloud.points).size:  # If the point cloud is not empty
        continue
    # Downsample for more speed
    downpcd = tile_cloud.voxel_down_sample(voxel_size=0.05)
    coords = np.asarray(downpcd.points)
    data = coords[:, 2]
    extreme_coords, extreme_data = locally_extreme_points(coords, data, 0.175)
    # Taking out some anomalies
    extreme_coords = np.asarray([i for i in np.asarray(extreme_coords) if i[2] > -4.5])

    if len(extreme_coords) > 5:
        extreme_points_pcd = o3d.geometry.PointCloud()
        extreme_points_pcd.points = o3d.utility.Vector3dVector(extreme_coords)
        extreme_points_pcd.paint_uniform_color([1.,0.,0.])

        bounding_boxes = create_bounding_boxes(extreme_points_pcd)

        for box in bounding_boxes:
            the_bounding_boxes.append(box)

        # TO GENERATE VISUALIZATIONS
        o3d.visualization.draw_geometries([tile_cloud, *bounding_boxes])

'''
final_tree_data = []
# TO GENERATE DATA FOR EACH TREE
count = 1
for t in the_bounding_boxes:
    features = extract_features(t, count)
    # Filter out some anomalies (arbritrary numbers - change depending on project)
    if (features[1] > 1.5 and features[1] < 4.2):
        final_tree_data.append(features)
        count+=1
        print(f"Tree {count} done.")
    
filename = 'tree_data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tree ID', 'Final Tree Height (m)', 'Final Crown Width (m)', 'Final Carbon Stock'])
    writer.writerows(final_tree_data)
'''

#o3d.visualization.draw_geometries([original_point_cloud, *the_bounding_boxes])
