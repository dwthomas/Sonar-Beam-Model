#! /usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon
import utils

def unit(v):
    if np.linalg.norm(v) == 0:
        return 0*v
    return v / np.linalg.norm(v)

def survey_line_approx(line, m):
    plan_m, to_wgs84 = utils.line_to_metric_crs(line)
    step = 1000  # meters
    gons = []
    
    line_segments = []
    for start, end in zip(plan_m.coords, plan_m.coords[1:]):
        line_segments.append(LineString([start, end]))
    for line_m in line_segments:
        length = line_m.length
        distances = np.arange(0, length + step, step)
        for i in range(1,len(distances)):
            s_prior = distances[i-1]
            p_prior = line_m.interpolate(s_prior)
            s = distances[i]
            p = line_m.interpolate(s)
            segment = gpd.GeoDataFrame(geometry = [LineString([p_prior, p])], crs = utils.metric_crs)
            mid = segment.interpolate(0.5, normalized=True)
            width = m.width_at(Point(*to_wgs84.transform(mid.x, mid.y)))
            potential_beam = segment.buffer(width/2, cap_style=2)
            gons.append(potential_beam)
    the_beam =  gpd.GeoDataFrame(geometry = pd.concat(gons), crs = utils.metric_crs).union_all()
    the_beam = gpd.GeoDataFrame(geometry = [the_beam], crs = utils.metric_crs)
    return the_beam.to_crs(utils.wgs84)

def survey_line(line, m):
    line_m, to_wgs84 = utils.line_to_metric_crs(line)

    step = 1000  # meters
    length = line_m.length
    distances = np.arange(0, length + step, step)

    
    
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
            w = m.width_at(p_wgs84) / 2.0   # meters
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
    poly_wgs84 = gpd.GeoSeries([poly_m], crs=utils.metric_crs).to_crs(utils.wgs84).iloc[0]
    poly_gdf = gpd.GeoDataFrame(geometry=[poly_wgs84], crs=utils.wgs84)

    lefts = gpd.GeoDataFrame(geometry = [LineString(list(left_pts))], crs = utils.metric_crs).to_crs(utils.wgs84)
    rights = gpd.GeoDataFrame(geometry = [LineString(list(right_pts))], crs = utils.metric_crs).to_crs(utils.wgs84)
    return poly_gdf, lefts, rights

if __name__ == "__main__":
    import argparse
    import sys
    from shapely import wkt

    parser = argparse.ArgumentParser(description="Sonar Beam Model")

    parser.add_argument("--method", choices = ["step", "box"], default = "step", help="Method for constructing the beam polygon. Default: step.")
    parser.add_argument(
        "--crs",
        default=utils.wgs84,
        help="Input CRS string. Default: wgs84.",
    )
    parser.add_argument(
        "--gebco-dir",
        type=utils.existing_dir,
        default="gebco_raster/",
        help="Path to folder containing the GEBCO dataset (must exist). Default: gebco_raster/",
    )
    parser.add_argument(
        "--extinction",
        type=str,
        default="EM302nautilus.txt",
        help='Extinction curve filename or comma-separated extinction curve. Default: EM302nautilus.txt\nExample: --extinction EM302nautilus.txt or --extinction "0.0 5.6,1608.0 6.6,3000.0 3.133,4000.0 2.205,5000.0 1.644,6000.0 1.198, ..."'
    )
    args = parser.parse_args()
    
    raw = sys.stdin.read().strip()
    geom = wkt.loads(raw)
    line_gdf = gpd.GeoDataFrame(geometry=[geom], crs=args.crs)

    beam = utils.load_beam(args.extinction)
    max_width = utils.max_width(beam)

    gebco_folder = args.gebco_dir
    envelope = utils.line_to_ellipse(line_gdf, width=max_width*1.1, resolution = 4)  # Example width of 100 km+
    m = utils.Map(envelope, gebco_folder, extinction_file=args.extinction)
    beam_polygon = None
    if args.method == "step":
        beam_polygon, lefts, rights = survey_line(line_gdf, m)
    elif args.method == "box":
        beam_polygon = survey_line_approx(line_gdf, m)
    print(beam_polygon.geometry.iloc[0].wkt)