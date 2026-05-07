import numpy as np
import geopandas as gpd
import shapely
from pathlib import Path

import os
from pyproj import Geod, CRS
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon, shape, box, mapping
from shapely.ops import unary_union, snap, linemerge
import time
import subprocess
import glob
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds
from rasterio.features import shapes
from rasterio.transform import xy, rowcol
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask

import pandas as pd
import manifold3d as m3d

import trimesh
import manifold3d as m3d
import pyvista as pv


wgs84 = CRS.from_epsg(4326)
web_mercator = CRS.from_epsg(3857)
metric_crs = web_mercator

land_buffer_width = 3000

def latlon_to_xy_displacement(lons, lats, lon0, lat0):
    # Create an azimuthal equidistant projection centered on your origin point
    aeqd_crs = CRS.from_proj4(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m')
    transformer = Transformer.from_crs("EPSG:4326", aeqd_crs, always_xy=True)
    
    xs, ys = transformer.transform(lons, lats)  # returns meters from origin
    return xs, ys


def xy_displacement_to_latlon(xs, ys, lon0, lat0):
    aeqd_crs = CRS.from_proj4(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m')
    transformer = Transformer.from_crs(aeqd_crs, "EPSG:4326", always_xy=True)
    
    lons, lats = transformer.transform(xs, ys)
    return lons, lats


def dem_to_structuredgrid(depth_grid, transform, lon0, lat0):
    """
    depth_grid: 2D numpy array from rasterio (rows, cols)
    transform: Affine transform from rasterio dataset.transform
    """
    
    nrows, ncols = depth_grid.shape
    
    # Generate pixel coordinates (col, row) for each cell
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))


    
    
    # Convert pixel coords to spatial coords using the affine transform
    xs, ys = transform * (cols, rows)

    # print(xs, ys)
    xs, ys = latlon_to_xy_displacement(xs, ys, lon0, lat0)
    
    zs = depth_grid  # use depth values as Z (negate if depths are positive-down)
    # print(zs.min(), zs.max()) 

    # PyVista StructuredGrid expects (X, Y, Z) each of shape (nrows, ncols)
    grid = pv.StructuredGrid(xs, ys, zs)
    # print(grid.dimensions)   # e.g. (500, 300, 1)
    # print(grid.points.shape) # e.g. (150000, 3)
    # print(xs.shape) 
    
    return grid

def existing_dir(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path_str}")
    return str(path)

def line_to_metric_crs(line, metric_crs = metric_crs):
    gdf_m = line.to_crs(metric_crs)
    plan_m = gdf_m.geometry.iloc[0]
    to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    return plan_m, to_wgs84

def remove_holes(polygon):
    if polygon.is_valid:
        # Recreate the polygon with no holes (interiors)
        return Polygon(polygon.exterior)
    return polygon

def shrink_polygon(polygon, distance):
    """Shrinks a polygon by offsetting its exterior ring inward.
    
    - Uses `unary_union` to merge fragmented pieces after shrinking.
    - Handles MultiLineString by constructing a valid MultiPolygon.
    - Preserves interior holes where possible.
    """
    
    # Offset the exterior ring inward
    offset_ext = polygon.exterior.offset_curve(-distance)

    # If offsetting results in a MultiLineString, try to create multiple polygons
    if isinstance(offset_ext, MultiLineString):
        polygons = [Polygon(line) for line in offset_ext.geoms if line.is_ring]
    else:
        polygons = [Polygon(offset_ext)] if offset_ext.is_ring else []

    # Preserve interior holes from the original polygon
    for poly in polygons:
        poly = Polygon(poly.exterior, holes=[hole for hole in polygon.interiors if poly.contains(hole)])

    # Merge all resulting polygons into a single geometry
    result = unary_union(polygons)

    return result if not result.is_empty else polygon  # Return original if shrinking failed


def load_beam(extinction_file):
    beam = None
    if os.path.isfile(extinction_file):
        beam = pd.read_csv(extinction_file, sep = r"\s+", header=None, names= ["depth", "extinction"], index_col = "depth")
    else:
        extinction = []
        for pair in extinction_file.split(","):
            depth, ext = pair.strip().split()
            extinction.append((float(depth), float(ext)))
        beam = pd.DataFrame(extinction, columns=["depth", "extinction"]).set_index("depth")
    return beam

def max_width(beam):
    m_width = 0
    for i in range(len(beam)):
        depth = beam.index[i]
        extinction = beam["extinction"].iloc[i]
        width = depth * extinction
        if np.isfinite(width) and (width > m_width):
            m_width = width
    return m_width

def p2vgp(p):
    return vg.Point(p[0], p[1])

def isend(edge, point):
    return (edge.p1 == point) or (edge.p2 == point)
sea_wkt = """PROJCS["ProjWiz_Custom_Equidistant_Cylindrical",
 GEOGCS["GCS_WGS_1984",
  DATUM["D_WGS_1984",
   SPHEROID["WGS_1984",6378137.0,298.257223563]],
  PRIMEM["Greenwich",0.0],
  UNIT["Degree",0.0174532925199433]],
 PROJECTION["Equidistant_Cylindrical"],
 PARAMETER["False_Easting",0.0],
 PARAMETER["False_Northing",0.0],
 PARAMETER["Central_Meridian",130.078125],
 PARAMETER["Standard_Parallel_1",25.8090083],
 UNIT["Meter",1.0]]"""
sea_crs = CRS.from_wkt(sea_wkt)

#intersects = guc[guc.intersects(gdf_path_line.geometry.iloc[0]).any()]
def sci_utility(intersections):
    return intersections.to_crs('ESRI:54009').length.sum()/1000

def length(gdf_path_line):
    return gdf_path_line.to_crs('ESRI:54009').length.sum()/1000

def get_verts(point_on_edge, polygon):
    for i in range(len(polygon.exterior.coords) - 1):
        edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])
        if edge.distance(point_on_edge) < 10:
            return polygon.exterior.coords[i], polygon.exterior.coords[i + 1]
    return None

def get_edges(point_on_edge, polygon, stop_point):
    start = None
    for i in range(len(polygon.exterior.coords) - 1):
        edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])
        if edge.distance(point_on_edge) < 10:
            start = i
            break #= polygon.exterior.coords[i], polygon.exterior.coords[i + 1]
    forward_edges = []
    for i in range(len(polygon.exterior.coords) - 1):
        index = (start + i) % len(polygon.exterior.coords)
        next_i = (start + i + 1) % len(polygon.exterior.coords)
        edge = LineString([polygon.exterior.coords[index], polygon.exterior.coords[next_i]])
        if i == 0:
            edge = LineString([point_on_edge, polygon.exterior.coords[next_i]])
        if edge.distance(stop_point) < 10:
            forward_edges.append(LineString([polygon.exterior.coords[index], stop_point]))
            break
        else:
            forward_edges.append(edge)
    reverse_edges = []
    for i in range(len(polygon.exterior.coords) - 1, 0, -1):
        index = (start + i) % len(polygon.exterior.coords)
        next_i = (start + i - 1) % len(polygon.exterior.coords)
        edge = LineString([polygon.exterior.coords[index], polygon.exterior.coords[next_i]])
        if edge.distance(stop_point) < 10:
            reverse_edges.append(LineString([polygon.exterior.coords[index], stop_point]))
            break
        else:
            reverse_edges.append(edge)
    return forward_edges, reverse_edges

def combine_almost_continuous_lines(multi_line, tolerance=10):
    if isinstance(multi_line, LineString):
        return multi_line  # Already a LineString

    if not isinstance(multi_line, MultiLineString):
        raise ValueError("Input must be a LineString or MultiLineString")

    # Step 1: Snap lines to close small gaps
    snapped = snap(multi_line, multi_line, tolerance)

    # print(snapped)

    # Step 2: Merge snapped lines
    merged = unary_union(snapped)

    # print(merged)
    
    # Step 3: Ensure we produced a single LineString
    if isinstance(merged, LineString):
        return merged

    # Step 4: Attempt to order and merge segments
    if isinstance(merged, MultiLineString):
        lines = list(merged.geoms)
        ordered_lines = [lines.pop(0)]

        while lines:
            current = ordered_lines[-1]
            for i, line in enumerate(lines):
                # Check if the current line connects to any other line
                if current.coords[-1] == line.coords[0]:
                    ordered_lines.append(lines.pop(i))
                    break
                elif current.coords[-1] == line.coords[-1]:
                    ordered_lines.append(LineString(line.coords[::-1]))
                    lines.pop(i)
                    break
                elif current.coords[0] == line.coords[-1]:
                    ordered_lines.insert(0, lines.pop(i))
                    break
                elif current.coords[0] == line.coords[0]:
                    ordered_lines.insert(0, LineString(line.coords[::-1]))
                    lines.pop(i)
                    break
            else:
                break  # No more connections found

        # Combine ordered lines if all are connected
        if len(lines) == 0:
            return LineString([pt for line in ordered_lines for pt in line.coords])

    raise ValueError("Cannot combine lines into a single continuous LineString")


def transform_coords(coords, lon0, lat0):
    xs, ys = zip(*coords)
    lons, lats = xy_displacement_to_latlon(np.array(xs), np.array(ys), lon0, lat0)
    return list(zip(lons, lats))

def transform_polygon(poly, lon0, lat0):
    exterior = transform_coords(poly.exterior.coords, lon0, lat0)
    interiors = [transform_coords(ring.coords, lon0, lat0) for ring in poly.interiors]
    return Polygon(exterior, interiors)

def polydata_to_shapely(mesh, lon0, lat0):
    # Get points projected to XY plane
    mesh = mesh.triangulate()
    points_2d = mesh.points[:, :2]  # drop Z

    # Extract triangles from faces
    # pyvista face array is [n_verts, i, j, k, n_verts, i, j, k, ...]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # strip the leading "3"

    # Build a shapely polygon for each triangle
    triangles = []
    for tri in faces:
        coords = points_2d[tri]
        poly = Polygon(coords)
        if poly.is_valid and not poly.is_empty:
            triangles.append(poly)

    # Merge all triangles into a single shape
    combined = unary_union(triangles)

    # Transform coordinates back to lat/lon


    if isinstance(combined, MultiPolygon):
        combined = MultiPolygon([transform_polygon(p, lon0, lat0) for p in combined.geoms])
    else:
        combined = transform_polygon(combined, lon0, lat0)
        
    return combined


def get_pos(lat, lng):
    return lat, lng


def _unwrap_lon_pair(lon1, lon2):
    """Return lon2 shifted by +/-360 so it is closest to lon1."""
    delta = lon2 - lon1
    if delta > 180:
        return lon2 - 360
    if delta < -180:
        return lon2 + 360
    return lon2


def _normalize_lon(lon):
    return ((lon + 180) % 360) - 180


def _closest_lon_to_ref(lon, ref_lon):
    """Shift lon by k*360 so it is closest to ref_lon."""
    return lon + 360 * round((ref_lon - lon) / 360)


def _lon_ranges_overlap(lon1_min, lon1_max, lon2_min, lon2_max):
    """Check if two longitude ranges overlap, accounting for dateline wrapping.
    
    A longitude range wraps if min > max (e.g., 170 to -170 wraps across ±180).
    Two ranges overlap if they share any longitude values on the globe.
    """
    # Normalize: if a range is wrapped (min > max), split into two normal ranges
    # and check if either part overlaps.
    
    r1_wraps = lon1_min > lon1_max
    r2_wraps = lon2_min > lon2_max
    
    if not r1_wraps and not r2_wraps:
        # Neither wraps: simple overlap test
        return lon1_min < lon2_max and lon1_max > lon2_min
    
    if r1_wraps and not r2_wraps:
        # r1 wraps (e.g., 170 to -170): splits into [170, 180] and [-180, -170]
        # r2 is normal: check if r2 overlaps either part
        return (lon2_min < 180 and lon2_max > lon1_min) or (lon2_min < lon1_max and lon2_max > -180)
    
    if not r1_wraps and r2_wraps:
        # r2 wraps: splits into [lon2_min, 180] and [-180, lon2_max]
        # r1 is normal: check if r1 overlaps either part
        return (lon1_min < 180 and lon1_max > lon2_min) or (lon1_min < lon2_max and lon1_max > -180)
    
    # Both wrap: both split across dateline, so they definitely overlap
    return True


def line_to_ellipse(line, width, metric_crs = web_mercator, resolution=64):
    """
    Constructs a GeoDataFrame of ellipses where each ellipse has the two points 
    from each segment of the line as the foci, and the sum of distances that 
    defines the ellipse is equal to the width. The ellipses are rotated to align 
    with the line segments.

    Parameters:
        line (LineString): A LineString representing the line.
        width (float): The sum of distances (major axis length) for the ellipses.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the ellipses as geometries.
    """
    ellipses = []

    # Work in geographic coordinates first so each segment can be unwrapped
    # across the antimeridian before local metric projection.
    line_wgs84 = line.to_crs(wgs84)
    coords = list(line_wgs84.geometry.iloc[0].coords)

    # Iterate through each segment of the line
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2_raw, lat2 = coords[i + 1]

        # If a segment crosses the dateline, shift endpoint longitude so
        # geometric operations see the short arc rather than a seam jump.
        lon2 = _unwrap_lon_pair(lon1, lon2_raw)

        center_lon = (lon1 + lon2) / 2.0
        center_lat = (lat1 + lat2) / 2.0
        center_lon_norm = _normalize_lon(center_lon)

        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon_norm} +datum=WGS84 +units=m"
        )
        to_metric = Transformer.from_crs(wgs84, local_crs, always_xy=True)
        to_wgs = Transformer.from_crs(local_crs, wgs84, always_xy=True)

        x1, y1 = to_metric.transform(lon1, lat1)
        x2, y2 = to_metric.transform(lon2, lat2)
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)

        # Calculate the center of the ellipse
        center = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

        # Calculate the distance between the two foci
        foci_distance = p1.distance(p2)

        # Calculate ellipse axes from focal distance and path width
        semi_major = (width + foci_distance) / 2
        semi_minor_sq = semi_major**2 - (foci_distance / 2) ** 2
        semi_minor = np.sqrt(max(semi_minor_sq, 0.0))

        # Create a unit circle and scale it to the ellipse dimensions
        ellipse = shapely.affinity.scale(
            center.buffer(1, resolution=resolution), xfact=semi_major, yfact=semi_minor
        )

        # Calculate segment bearing in local metric plane and rotate ellipse
        angle = np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))
        ellipse = shapely.affinity.rotate(ellipse, angle, origin="center")

        # Transform ellipse coordinates back to lon/lat. Keep longitudes near
        # the segment center to avoid reintroducing seam jumps.
        lonlat_coords = []
        for x, y in ellipse.exterior.coords:
            lon, lat = to_wgs.transform(x, y)
            lon_adj = _closest_lon_to_ref(lon, center_lon)
            lonlat_coords.append((lon_adj, lat))

        ellipses.append(Polygon(lonlat_coords))

    gdf = gpd.GeoDataFrame(geometry=ellipses, crs=wgs84)
    return gdf.to_crs(line.crs)

def load_gebco_region(tile_paths: list[str], polygon):
    """
    Args:
        tile_paths: List of paths to all GEBCO .tif files
        polygon: A Shapely geometry (Polygon or MultiPolygon) in WGS84
    """
    # print(type(polygon))
    polygon = polygon.geometry.union_all()
    bbox = polygon.bounds  # (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = bbox

    # Find overlapping tiles using bounding box (fast), with dateline safety
    overlapping = []

    for path in tile_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            # Check latitude overlap (simple, no wrapping)
            if b.bottom < max_lat and b.top > min_lat:
                # Check longitude overlap accounting for dateline wrapping
                if _lon_ranges_overlap(b.left, b.right, min_lon, max_lon):
                    overlapping.append(path)

    if not overlapping:
        raise ValueError("No GEBCO tiles overlap the requested polygon.")

    # print(f"Found {len(overlapping)} overlapping tile(s).")

    # Convert polygon to GeoJSON-like dict for rasterio
    geom = [mapping(polygon)]

    if len(overlapping) == 1:
        with rasterio.open(overlapping[0]) as src:
            data, transform = rio_mask(src, geom, crop=True, nodata=-1)
            crs = src.crs
    else:
        # Mosaic tiles first, then mask to polygon
        datasets = [rasterio.open(p) for p in overlapping]
        try:
            mosaic, mosaic_transform = merge(
                datasets,
                bounds=(min_lon, min_lat, max_lon, max_lat),
            )
            crs = datasets[0].crs
        finally:
            for ds in datasets:
                ds.close()

        # Write mosaic to a memory file, then apply polygon mask
        from rasterio.io import MemoryFile
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                count=1,
                dtype=mosaic.dtype,
                crs=crs,
                transform=mosaic_transform,
            ) as mem_ds:
                mem_ds.write(mosaic)
                data, transform = rio_mask(mem_ds, geom, crop=True, nodata=-1)

    return data[0], transform, crs 


def load_raster(filename, bbox):
    with rasterio.open(filename) as ds:
        window = from_bounds(*bbox, transform=ds.transform)
        crs = ds.crs
        transform = rasterio.windows.transform(window, ds.transform)
        return ds.read(1, window=window), transform, crs


class Map:
    def __init__(self, mask, gebco_folder, extinction_file = "EM302nautilus.txt"):
        self.beam = load_beam(extinction_file)
        
        tid_files = glob.glob(os.path.join(gebco_folder, "*_tid_*.tif"))
        self.tid_raster, self.tid_transform, self.tid_crs = load_gebco_region(tid_files, mask)
        depth_files = glob.glob(os.path.join(gebco_folder, "*_sub_ice_*.tif"))
        self.depth_raster, self.depth_transform, self.depth_crs = load_gebco_region(depth_files, mask)
        # self.land_raster = (255 * (self.depth_raster < 0)).astype(np.uint8)
        self.land_raster = (self.tid_raster == 0).astype(np.uint8)
        self.tid_raster = self.tid_raster.astype(np.uint8)
        self.unmapped_raster = (self.tid_raster != 11).astype(np.uint8)  # (self.tid_raster > 17) * (1 - self.land_raster)

        # print(self.depth_raster.shape)
        # bbox = (-180.0, 0, 180, 90)
        
        # left, bottom, right, top = bbox
        # Polygonize land        
        land_polygons = []
        for geom, value in shapes(
            self.land_raster,
            mask=self.land_raster,
            transform=self.tid_transform
        ):
            if value == 1:
                land_polygons.append(shape(geom))
        # Create GeoDataFrame
        self.land_polygons = gpd.GeoDataFrame(geometry=land_polygons, crs=self.tid_crs)
        self.grow_land_polygons()

        #polygonize unmapped
        unmapped_polygons = []
        for geom, value in shapes(
            self.unmapped_raster,
            mask=self.unmapped_raster,
            transform=self.tid_transform
        ):
            if value == 1:
                unmapped_polygons.append(shape(geom))
        # Create GeoDataFrame
        self.unmapped_polygons  = gpd.GeoDataFrame(geometry=unmapped_polygons, crs=self.tid_crs)
        merged = self.unmapped_polygons.geometry.union_all()
        self.unmapped_polygons = gpd.GeoDataFrame(
            geometry=[merged], crs=self.unmapped_polygons.crs
        ).explode(index_parts=False, ignore_index=True)
        self.shrink_unmapped_polygons()

    def polygonize_seafloor(self, nodata=None):
        """Create a PyVista StructuredGrid seafloor surface in ``metric_crs``.

        The grid is built on raster corner vertices. Corner z-values are computed
        by averaging all valid neighboring raster-cell depths.
        """
        import pyvista as pv

        depth = self.depth_raster.astype(float)
        valid = np.isfinite(depth)
        if nodata is not None:
            valid &= depth != nodata

        nrows, ncols = depth.shape

        # Average surrounding cell depths onto corner vertices.
        z_sum = np.zeros((nrows + 1, ncols + 1), dtype=float)
        z_cnt = np.zeros((nrows + 1, ncols + 1), dtype=float)
        d = np.where(valid, depth, 0.0)
        c = valid.astype(float)

        z_sum[0:nrows, 0:ncols] += d
        z_sum[1:nrows + 1, 0:ncols] += d
        z_sum[0:nrows, 1:ncols + 1] += d
        z_sum[1:nrows + 1, 1:ncols + 1] += d

        z_cnt[0:nrows, 0:ncols] += c
        z_cnt[1:nrows + 1, 0:ncols] += c
        z_cnt[0:nrows, 1:ncols + 1] += c
        z_cnt[1:nrows + 1, 1:ncols + 1] += c

        z = np.full((nrows + 1, ncols + 1), np.nan, dtype=float)
        np.divide(z_sum, z_cnt, out=z, where=z_cnt > 0)

        # Build corner coordinate grid in source CRS, then project to metric CRS.
        col_idx, row_idx = np.meshgrid(
            np.arange(ncols + 1, dtype=float),
            np.arange(nrows + 1, dtype=float),
        )
        a, b, c0, d0, e, f0 = self.depth_transform[:6]
        x_src = a * col_idx + b * row_idx + c0
        y_src = d0 * col_idx + e * row_idx + f0

        to_metric = Transformer.from_crs(self.depth_crs, metric_crs, always_xy=True)
        x_m, y_m = to_metric.transform(x_src.ravel(), y_src.ravel())
        x_m = np.asarray(x_m).reshape(z.shape)
        y_m = np.asarray(y_m).reshape(z.shape)

        grid = pv.StructuredGrid(x_m, y_m, z)
        grid["depth"] = z.ravel(order="F")
        return grid
        
    def center_coords(self):
        rows, cols = np.meshgrid(
            np.arange(self.depth_raster.shape[0]),
            np.arange(self.depth_raster.shape[1]),
            indexing='ij'
        )
        lons, lats = rasterio.transform.xy(self.depth_transform, rows, cols)
        return lons, lats

    def ul_coords(self):
        rows, cols = np.meshgrid(
            np.arange(self.depth_raster.shape[0]),
            np.arange(self.depth_raster.shape[1]),
            indexing='ij'
        )
        lons, lats = rasterio.transform.xy(self.depth_transform, rows, cols, offset='ul')
        return lons, lats

    def plot_mapped_raster(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.imshow(self.unmapped_raster, cmap="Blues_r")
        plt.colorbar(label="TID")
        plt.title("GEBCO 2025 Bathymetry")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    def plot_min_raster(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.min_depth_raster, cmap="Blues_r")
        plt.colorbar(label="Elevation")
        plt.title("GEBCO 2008 Bathymetry")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")


    def plot_land(self, m = None):
        import folium
        if not m:
            m = folium.Map(location=[lat1, lon1],  zoom_start=8, tiles="Esri.OceanBasemap")

        # Add land polygons
        folium.GeoJson(
            self.land_polygons,
            name="Land",
            style_function=lambda x: {
                "fillColor": "#2ecc71",   # green land
                "color": "#145a32",       # border
                "weight": 1,
                "fillOpacity": 0.6,
            },
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m

    def seafloor_mesh(self, debug=False, make_solid=True, bottom_z=None, use_relative_meters=False):
        """
        Build a Manifold mesh from seafloor raster data.

        Manifold requires a closed (watertight) 2-manifold surface. A raw terrain
        sheet is open, so by default this function creates a thin closed solid
        using the seafloor as the top surface.

        If bottom_z is not provided, the solid bottom is set to twice the depth
        of Earth's deepest ocean point (-10,935 m), i.e. -21,870 m.

        Set use_relative_meters=True to convert horizontal coordinates to meters
        in metric_crs and translate them so the top-left raster point is (0, 0).
        """
        lon, lat = self.center_coords()
        depth = np.asarray(self.depth_raster)
        H, W = depth.shape

        if debug:
            print("seafloor_mesh shapes:", "lon", lon.shape, "lat", lat.shape, "depth", depth.shape)
            print("seafloor_mesh dtypes:", "lon", lon.dtype, "lat", lat.dtype, "depth", depth.dtype)

        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if lon.shape == depth.shape and lat.shape == depth.shape:
            lon_flat = lon.reshape(-1)
            lat_flat = lat.reshape(-1)
        elif lon.ndim == 1 and lat.ndim == 1 and lon.shape == lat.shape and lon.size == depth.size:
            lon_flat = lon
            lat_flat = lat
        else:
            raise ValueError(
                f"Shape mismatch: lon={lon.shape}, lat={lat.shape}, depth={depth.shape}. "
                "lon/lat must either match depth.shape or be flattened to depth.size."
            )

        if use_relative_meters:
            to_metric = Transformer.from_crs(self.depth_crs, metric_crs, always_xy=True)
            x_flat, y_flat = to_metric.transform(lon_flat, lat_flat)
            x_flat = np.asarray(x_flat, dtype=np.float64)
            y_flat = np.asarray(y_flat, dtype=np.float64)

            x0_src, y0_src = rasterio.transform.xy(self.depth_transform, 0, 0, offset="center")
            x0, y0 = to_metric.transform(x0_src, y0_src)
            x_flat = x_flat - float(x0)
            y_flat = y_flat - float(y0)
        else:
            x_flat = lon_flat
            y_flat = lat_flat

        depth_flat = depth.reshape(-1)
        finite = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(depth_flat)
        if not np.all(finite):
            bad = np.count_nonzero(~finite)
            raise ValueError(
                f"Seafloor mesh contains {bad} non-finite vertices. "
                "Check the raster crop, nodata handling, and coordinate transform."
            )

        # --- Vertices ---
        # Stack into (H*W, 3) array of [x, y, depth]
        verts = np.column_stack([
            x_flat,
            y_flat,
            depth_flat
        ]).astype(np.float32)

        if debug:
            print("verts:", verts.shape, "min/max depth:", float(np.min(depth)), float(np.max(depth)))
            if use_relative_meters:
                print("relative meters x/y min/max:",
                      (float(np.min(x_flat)), float(np.max(x_flat))),
                      (float(np.min(y_flat)), float(np.max(y_flat))))

        # --- Faces (triangles) ---
        # For each grid cell (i,j), create 2 triangles:
        #   (i,j) --- (i,j+1)
        #     |  \       |
        #   (i+1,j) - (i+1,j+1)
        def idx(i, j):
            return i * W + j

        i, j = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing='ij')
        i, j = i.flatten(), j.flatten()

        tri1 = np.column_stack([idx(i, j),     idx(i+1, j), idx(i+1, j+1)])
        tri2 = np.column_stack([idx(i, j),     idx(i+1, j+1), idx(i,   j+1)])
        faces = np.vstack([tri1, tri2]).astype(np.uint32)

        if debug:
            print("faces:", faces.shape, "expected triangles:", 2 * (H - 1) * (W - 1))

        if make_solid:
            n_top = verts.shape[0]
            if bottom_z is None:
                bottom_z = 2.0 * (-10935.0)
            bottom_z = float(bottom_z)

            # Bottom cap vertices directly below top vertices.
            bottom_verts = verts.copy()
            bottom_verts[:, 2] = bottom_z

            # Top + bottom vertices.
            verts = np.vstack([verts, bottom_verts]).astype(np.float32)

            # Bottom cap triangles are top triangles with reversed winding.
            bottom_faces = (faces[:, [0, 2, 1]] + n_top).astype(np.uint32)

            # Side walls around outer raster perimeter.
            perimeter = []
            perimeter.extend([idx(0, c) for c in range(W)])
            perimeter.extend([idx(r, W - 1) for r in range(1, H)])
            perimeter.extend([idx(H - 1, c) for c in range(W - 2, -1, -1)])
            perimeter.extend([idx(r, 0) for r in range(H - 2, 0, -1)])

            side_tris = []
            for k in range(len(perimeter)):
                t0 = perimeter[k]
                t1 = perimeter[(k + 1) % len(perimeter)]
                b0 = t0 + n_top
                b1 = t1 + n_top
                side_tris.append([t0, t1, b1])
                side_tris.append([t0, b1, b0])

            side_faces = np.asarray(side_tris, dtype=np.uint32)
            faces = np.vstack([faces, bottom_faces, side_faces]).astype(np.uint32)

            if debug:
                print("solid verts:", verts.shape)
                print("solid faces:", faces.shape)
                print("bottom_z:", bottom_z)

        # --- Build Manifold mesh ---
        mesh = m3d.Mesh(vert_properties=verts, tri_verts=faces)
        manifold = m3d.Manifold(mesh=mesh)

        if debug:
            try:
                print("manifold empty:", manifold.is_empty())
            except Exception:
                pass

        return manifold
            

    def plot_unmapped(self, m = None):
        import folium
        if not m:
            m = folium.Map( zoom_start=8, tiles="Esri.OceanBasemap")
        folium.GeoJson(
            self.unmapped_polygons,
            style_function=lambda feature: {
                "weight": 1,
                "fillOpacity": 0.6,
            },
        ).add_to(m)
        return m

    def shrink_unmapped_polygons(self):
        shrunk = []
        for pgon in self.unmapped_polygons.geometry:
            centroid = pgon.centroid
            # print(centroid)
            beam_width = self.width_at(centroid)
            pgon_merc = gpd.GeoSeries([pgon], crs=self.unmapped_polygons.crs)
            pgon_merc = pgon_merc.simplify(.005)

            pgon_merc = pgon_merc.to_crs(web_mercator).iloc[0]
            new_gon_merc = pgon_merc.buffer(-beam_width / 2)
            if new_gon_merc.area > 0:
                new_gon = gpd.GeoSeries([new_gon_merc], crs=web_mercator).to_crs(self.unmapped_polygons.crs).iloc[0]
                shrunk.append(new_gon)

        self.unmapped_polygons = gpd.GeoDataFrame(geometry=shrunk, crs=self.unmapped_polygons.crs)

    def grow_land_polygons(self):
        grown = []
        for pgon in self.land_polygons.geometry:
            centroid = pgon.centroid
            
            pgon_merc = gpd.GeoSeries([pgon], crs=self.land_polygons.crs)

            pgon_merc = pgon_merc.to_crs(web_mercator).iloc[0]
            new_gon_merc = pgon_merc.buffer(land_buffer_width)
            if new_gon_merc.area > 0:
                new_gon = gpd.GeoSeries([new_gon_merc], crs=web_mercator).to_crs(self.land_polygons.crs).iloc[0]
                new_gon = new_gon.simplify(.005)
                grown.append(new_gon)

        self.land_polygons = gpd.GeoDataFrame(geometry=grown, crs=self.land_polygons.crs)


    def index_of(self, point):
        transformer = Transformer.from_crs(
            "EPSG:4326",
            self.depth_crs,
            always_xy=True
        )

        x, y = transformer.transform(point.x, point.y)

        # Convert to row/col in window
        row, col = rowcol(self.depth_transform, x, y)
        return int(col), int(row)

    def coords_of(self, col, row):
        """Return (lat, lon) for the center of a depth-raster pixel."""
        col = int(col)
        row = int(row)

        nrows, ncols = self.depth_raster.shape
        if row < 0 or row >= nrows or col < 0 or col >= ncols:
            raise IndexError(f"Pixel out of bounds: row={row}, col={col}, shape={self.depth_raster.shape}")

        x, y = rasterio.transform.xy(self.depth_transform, row, col, offset="center")
        to_wgs84 = Transformer.from_crs(self.depth_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(x, y)
        return lat, lon

    
    

    def in_radius_of(self, point, radius):
        col, row = self.index_of(point)
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        
        x1, y1 = rasterio.transform.xy(self.depth_transform, row, col)
        x1_m, y1_m = transformer.transform(x1, y1)

        points = [(col, row)]
        for i in [1, -1]: # up/down
            for j in [1, -1]: # left/right
                y = 1
                while True:
                    r = row + i*y
                    gotone = False
                    x = 1
                    while True:
                        c = col + j*x
                        x2, y2 = rasterio.transform.xy(self.depth_transform, r, c)
                        x2_m, y2_m = transformer.transform(x2, y2)

                        dist = np.sqrt((x2_m - x1_m)**2 + (y2_m - y1_m)**2)
                        if dist > radius:
                            break
                        gotone = True
                        points.append((c, r))
                        x+=1
                    if not gotone:
                        break
                    y += 1
        return points


        

    def depth_at(self, point):
        col, row = self.index_of(point)
        # print(row, col, self.depth_raster.shape)
        if row >= self.depth_raster.shape[0]:
            row = self.depth_raster.shape[0]-1
        if col >= self.depth_raster.shape[1]:
            col = self.depth_raster.shape[1]-1
        # print(col, row)
        # print(self.depth_raster.shape)
        # print( self.depth_raster[row, col])
        return self.depth_raster[row, col]

    def width_at_depth(self, depth):
        depth= -depth
        s = self.beam["extinction"]
        s2 = s.reindex(s.index.union([depth])).sort_index()   
        extinction = s2.interpolate(method='index').loc[depth]
        result = depth * extinction
        if not np.isfinite(result):
            return 0.0
            raise ValueError(f"Non-finite width calculated at point {point}. Depth: {depth}, Extinction: {extinction}")
        return result

    def width_at(self, point):
        depth = self.depth_at(point)
        return self.width_at_depth(depth)


    def survey_line(self, line):
        gdf_m = line.to_crs(metric_crs)
        line_m = gdf_m.geometry.iloc[0]
        to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

        step = 1000  # meters
        length = line_m.length
        distances = np.arange(0, length + step, step)

        def unit(v):
            return v / np.linalg.norm(v)
        
        left_pts = []
        right_pts = []
        
        for s in distances:
            p = line_m.interpolate(s)
        
            # tangent via finite difference
            eps = 1.0
            s1 = max(s - eps, 0)
            s2 = min(s + eps, length)
        
            p1 = line_m.interpolate(s1)
            p2 = line_m.interpolate(s2)
            # print(p, p1, p2)
            t = np.array([p2.x - p1.x, p2.y - p1.y])
            try:
                t = unit(t)
            except:
                continue
        
            # perpendicular normal
            n = np.array([-t[1], t[0]])
        
            # convert sample point → WGS84 for width()
            lon, lat = to_wgs84.transform(p.x, p.y)
            p_wgs84 = Point(lon, lat)
            try:
                w = self.width_at(p_wgs84) / 2.0   # meters
            except:
                continue
            left = Point(p.x + n[0]*w, p.y + n[1]*w)
            right = Point(p.x - n[0]*w, p.y - n[1]*w)

            if np.isfinite(left.x) and np.isfinite(left.y):
                left_pts.append(left)
            if np.isfinite(right.x) and np.isfinite(right.y):
                right_pts.append(right)
        
    
            # print(left, right)

        poly_m = Polygon(list(left_pts) + list(reversed(right_pts)))
        poly_wgs84 = gpd.GeoSeries([poly_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
        poly_gdf = gpd.GeoDataFrame(geometry=[poly_wgs84], crs="EPSG:4326")

        lefts = gpd.GeoDataFrame(geometry = [LineString(list(left_pts))], crs = metric_crs).to_crs("EPSG:4326")
        rights = gpd.GeoDataFrame(geometry = [LineString(list(right_pts))], crs = metric_crs).to_crs("EPSG:4326")
        return poly_gdf, lefts, rights

    def survey_line_3D(self, line):
        gdf_m = line.to_crs(metric_crs)
        line_m = gdf_m.geometry.iloc[0]
        to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
        
        lon0, lat0 = line.loc[0].geometry.coords[0]


        # create a structured grid mesh of the seafloor
        depth_grid = self.depth_raster.astype(float)
        depth_grid[depth_grid >= -2] = 2*np.nanmin(depth_grid.flatten())
        sg = dem_to_structuredgrid(depth_grid, self.depth_transform, lon0, lat0)
        seafloor_mesh = sg.extract_surface(algorithm ='dataset_surface')

        # create a 2D model of the sonar beam
        points1 = [(0, 0)]
        points2 = []
        for depth in self.beam.index:
            width = self.width_at_depth(-depth)
            points1.append((float(-width/2), -depth))
            points2.append((float(width/2), -depth))
        pgon = Polygon(points1 + list(reversed(points2)))

        # create a 3D model of the sonar beam by extruding the 2D model along the survey line
        line_lons, line_lats = zip(*line.geometry[0].coords)  # unpack lon, lat pairs

        line_xs, line_ys = latlon_to_xy_displacement(
            np.array(line_lons),
            np.array(line_lats),
            lon0, lat0
        )

        line_points = np.column_stack([line_xs, line_ys, np.zeros(len(line_xs))])
        mesh = trimesh.creation.sweep_polygon(pgon, line_points)
        mesh.vertices[:, 2] *= -1
        beam_pv = pv.wrap(mesh)
        tri_seafloor = seafloor_mesh.triangulate()

        # find their intersection, then project to latlon
        clipped = seafloor_mesh.clip_surface(beam_pv, invert=False)

        clipped_flattened = polydata_to_shapely(clipped, lon0, lat0)

        poly_gdf = gpd.GeoDataFrame(geometry = [clipped_flattened], crs = wgs84)
            
        return poly_gdf, None, None
    

def get_polys():
    if os.path.exists("glc_simp.feather") and os.path.exists("guc_simp.feather") and os.path.exists("glc_orig.feather"):
        glc_simp = gpd.read_feather("glc_simp.feather")
        guc_simp = gpd.read_feather("guc_simp.feather")
        glc_orig = gpd.read_feather("glc_orig.feather")
        return glc_simp, guc_simp, glc_orig, get_center(glc_simp.to_crs(epsg=4326))
    guc_simp = gpd.read_file("GebcoHICrop/guc.json")
    guc_simp = guc_simp.to_crs('ESRI:54009')
    ind = np.argsort(-guc_simp.geometry.area)
    x = guc_simp.iloc[ind]
    y_guc = guc_simp.iloc[ind]
    x.geometry = x.geometry.apply(remove_holes)
    x = x.buffer(-2000)
    guc_simp.geometry = x.simplify(1000)
    guc_simp = guc_simp[~guc_simp.geometry.is_empty]


    glc_orig = gpd.read_file("GebcoHICrop/merge-glc.json")
    glc_orig = glc_orig.to_crs('ESRI:54009')
    glc_simp = glc_orig.copy()
    ind = np.argsort(-glc_simp.geometry.area)
    x = glc_simp.iloc[ind]
    y = glc_simp.iloc[ind]
    x.geometry = x.geometry.apply(remove_holes)
    x.geometry = x.buffer(1000)
    x.geometry = x.simplify(1000)
    x.geometry = x.geometry.apply(lambda y: y.convex_hull)
    x = x.dissolve().explode(index_parts=False).reset_index(drop=True)
    x.geometry = x.simplify(1000)

    glc_simp = x
    glc_simp['geometry'] = glc_simp['geometry'].apply(remove_holes)
    glc_simp = glc_simp.loc[glc_simp.geometry.area > 50000000] # get rid of small areas
    glc_simp.to_feather("glc_simp.feather")
    glc_orig.to_feather("glc_orig.feather")
    guc_simp.to_feather("guc_simp.feather")
    return glc_simp.to_crs(epsg=4326), guc_simp.to_crs(epsg=4326), glc_orig.to_crs(epsg=4326), get_center(glc_simp.to_crs(epsg=4326))

def get_center(_glc_simp):
    center = _glc_simp.union_all().centroid
    return (np.asarray(center.coords[0])[::-1])