
import os
import cv2
import json
import numpy as np
from osgeo import osr
from osgeo import gdal
import image_utils

ROOT_DIR = '..'


def get_coordinates_from_json(coords_json_file):
    city_coords = []
    with open(coords_json_file) as f:
        contents = json.load(f)
        contents = contents['coordinates'][0]
        coordinates = np.asarray(contents)
        longi_left = np.min(coordinates[:, 0]).astype(np.float32) # longi_left
        longi_right = np.max(coordinates[:, 0]).astype(np.float32)  # longi_right
        lati_top = np.max(coordinates[:, 1]).astype(np.float32)  # lati_top
        lati_bottom = np.min(coordinates[:, 1]).astype(np.float32)  # lati_bottom
        city_coords.append(longi_left)
        city_coords.append(longi_right)
        city_coords.append(lati_top)
        city_coords.append(lati_bottom)
        return city_coords




def prepare_test_patches_given_meta_json():
    dataset_dir = os.path.join(ROOT_DIR, '..')
    meta_json_file = os.path.join(dataset_dir, 'patch_info.json')
    f = open(meta_json_file)
    patch_contents = json.load(f)
    f.close()
    mask_dir = os.path.join(dataset_dir, 'full')
    patch_save_dir = os.path.join(dataset_dir, 'patch')
    os.makedirs(patch_save_dir, exist_ok=True)
    # planet satellite
    img_names = ['blue', 'green', 'red', 'ndvi', 'nir', ]
    for image_name in img_names:
        image_utils.divide_planet_single_channel_images_into_patches(patch_contents, image_name, mask_dir,
                                                             patch_save_dir, spec_city=None)


def get_city_img_width_height():
    country = ''
    json_file_to_save = os.path.join(ROOT_DIR, 'us_satellite_size.json')
    record = {}
    us_cities = ['losangeles_230420', 'newyork_230601', 'seattle_230702']
    for city_date in us_cities:
        city, date = city_date.split('_')
        ndvi_img_file = os.path.join(ROOT_DIR, f'PlanetScope/USA/{city_date}/ndvi255.png')
        image = cv2.imread(ndvi_img_file, 0)
        sat_h, sat_w = image.shape
        record[city] = {}
        record[city]['width'] = sat_w
        record[city]['height'] = sat_h
    with open(json_file_to_save, 'w') as f:
        json.dump(record, f, indent=4)


# https://gdal.org/index.html
# https://stackoverflow.com/questions/33537599/how-do-i-write-create-a-geotiff-rgb-image-file-in-python
def create_geotiff_from_png():
    data_dir = os.path.join(ROOT_DIR, 'patches')
    result_dir = os.path.join(data_dir, '')
    coords_json_dir = os.path.join(ROOT_DIR, 'city_coordinates')
    patch_json_file = os.path.join(data_dir, 'US_final_patches.json')
    f = open(patch_json_file)
    patch_contents = json.load(f)
    f.close()
    for city in patch_contents.keys():
        if city == 'belfast':
            continue
        date = patch_contents[city]['dates'][0]
        coords_json_file = os.path.join(coords_json_dir, f'{city}_coordinates.json')
        img_file = os.path.join(result_dir, f'{city}_{date}_stitched_prediction.png')
        geotiff_to_save = os.path.join(result_dir, f'{city}_{date}_prediction.tif')
        city_coords = get_coordinates_from_json(coords_json_file)

        longis, latis = [city_coords[0], city_coords[1]], [city_coords[2], city_coords[3]]
        img = cv2.imread(img_file)
        height, width, _ = img.shape
        xmin, ymin, xmax, ymax = [min(longis), min(latis), max(longis), max(latis)]
        xres = (xmax - xmin) / float(width)
        yres = (ymax - ymin) / float(height)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)

        # create the 3-band raster file
        dst_ds = gdal.GetDriverByName('GTiff').Create(geotiff_to_save, xsize=width, ysize=height, bands=3,
                                                      eType=gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(3857)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(img[:, :, 2])  # write r-band to the raster
        dst_ds.GetRasterBand(2).WriteArray(img[:, :, 1])  # write g-band to the raster
        dst_ds.GetRasterBand(3).WriteArray(img[:, :, 0])  # write b-band to the raster
        dst_ds.FlushCache()  # write to disk
        dst_ds = None


if __name__ == "__main__":
    create_geotiff_from_png()