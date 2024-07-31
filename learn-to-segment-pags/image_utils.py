import os
import cv2
import json
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import numpy as np


def longi_lati_to_pixel_coords(longis, latis, sat_longi_l, sat_longi_r, sat_lati_t, sat_lati_b, sat_w, sat_h):
    xs = (longis - sat_longi_l) / (sat_longi_r - sat_longi_l) * sat_w
    ys = (latis - sat_lati_t) / (sat_lati_b - sat_lati_t) * sat_h
    xs = np.expand_dims(xs, axis=1)
    ys = np.expand_dims(ys, axis=1)
    return np.hstack((xs, ys)).astype(np.int32)


def pixel_coords_to_longi_lati(xs, ys, sat_longi_l, sat_longi_r, sat_lati_t, sat_lati_b, sat_w, sat_h):
    xs = np.expand_dims(xs, axis=1)
    ys = np.expand_dims(ys, axis=1)
    longis = xs * (sat_longi_r - sat_longi_l) / sat_w + sat_longi_l
    latis = ys * (sat_lati_b - sat_lati_t) / sat_h + sat_lati_t
    return np.hstack((longis, latis))


def reproject_satellite_image_crs_to_epsg(in_image_file, image_file_to_write, dst_crs='EPSG:3857'):
    '''
    :param in_image_file: input .tif image with raw data, can be any band combination
    :param image_file_to_write:
    :param dst_crs: destination CRS
    :return:
    '''
    tmp_img_file = os.path.join(os.path.dirname(image_file_to_write), 'tmp_img.tif')
    # re-projection, direct output image written as a temporary image
    with rasterio.open(in_image_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(tmp_img_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    # remove blank boundary and write final output image
    tmp_img = rasterio.open(tmp_img_file)
    profile = tmp_img.profile
    with rasterio.open(image_file_to_write, 'w', **profile) as dst:
        raster = tmp_img.read()
        image_g = np.mean(raster, axis=0)
        minimal_h = np.max(image_g, axis=1)
        minimal_w = np.max(image_g, axis=0)
        h_inds = np.flatnonzero(minimal_h)
        w_inds = np.flatnonzero(minimal_w)
        y0, y1 = h_inds[0], h_inds[-1]
        x0, x1 = w_inds[0], w_inds[-1]
        # PlanetScope images are in 16-bit
        img_data = raster[:, y0:y1+1, x0: x1+1].astype(rasterio.uint16)
        dst.write(img_data)
    print(f'Re-projected image written at: {image_file_to_write}')


# https://developers.planet.com/docs/planetschool/calculate-an-ndvi-in-python/
def calculate_ndvi_planetscope(composite_img, xml_meta, ndvi_image_to_write, chn_red, chn_nir):
    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    # https://developers.planet.com/docs/apis/data/sensors/
    src = rasterio.open(composite_img)
    band_red = src.read(chn_red)
    band_nir = src.read(chn_nir)
    from xml.dom import minidom
    xmldoc = minidom.parse(xml_meta)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        bn = int(bn)
        if bn in range(1, 9, 1):
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[bn] = float(value)
    # Multiply by corresponding coefficients
    band_red = band_red * coeffs[chn_red]
    band_nir = band_nir * coeffs[chn_nir]
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    numerator = band_nir + band_red
    min_val, max_val = np.min(numerator), np.max(numerator)
    print(f'NDVI: band_nir+band_red min_val:{min_val} max_val:{max_val}')
    np.clip(numerator, 0.001, max_val, out=numerator)
    ndvi = (band_nir.astype(float) - band_red.astype(float)) / numerator

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1)
    # Create the file
    with rasterio.open(ndvi_image_to_write, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))


def binarize_image(img, thresh=128, dtype='u1'):
    bin_img = np.zeros_like(img, dtype=dtype)
    bin_img = np.where(img > thresh, 255, bin_img)
    return bin_img


def divide_planet_single_channel_images_into_patches(patch_contents, image_name, image_dir, patch_save_dir, spec_city=None):
    for city, v in patch_contents.items():
        if spec_city is not None and city != spec_city:
            continue
        for date in v['dates']:
            # image_file = os.path.join(image_dir, f'{city}_{date}', image_name + '255.png')
            image_file = os.path.join(image_dir, f'{city}_{image_name}.png')  # US 3city test image
            if not os.path.isfile(image_file):
                print(f'file not exist: {image_file}')
                continue
            image_full = cv2.imread(image_file, 0)
            p_c = 0
            for patch in v['patches']:
                patch_count = patch['patch_count']
                # patch_file_to_save = os.path.join(patch_save_dir, f'{city}_{date}_{image_name}_patch{patch_count}.png')
                patch_file_to_save = os.path.join(patch_save_dir, f'{city}_{image_name}_patch{patch_count}.png')
                pixel_coordinates = patch['pixel_coordinate']
                x0, y0, x1, y1 = pixel_coordinates[0][0], pixel_coordinates[0][1], pixel_coordinates[1][0], pixel_coordinates[1][1]
                img_p = image_full[y0:y1, x0:x1]
                cv2.imwrite(patch_file_to_save, img_p)
                p_c += 1
            print(f'patch count: {p_c} for {city} {date} {image_name} ')


def divide_single_channel_masks_into_patches(patch_contents, image_name, image_root, patch_save_dir, spec_city=None):
    for city, v in patch_contents.items():
        if spec_city is not None and city != spec_city:
            continue
        image_file = os.path.join(image_root, f'{city}_{image_name}.png')
        # planet satellite images maybe without city name
        # image_file = os.path.join(image_root, f'{image_name}.png')
        patch_save_name = f'{city}_{image_name}'
        if not os.path.isfile(image_file):
            print(f'file not exist: {image_file}')
            continue
        image_full = cv2.imread(image_file, 0)
        image_full = binarize_image(image_full, thresh=50)
        p_c = 0
        # image_name = os.path.splitext(os.path.basename(image_file))[0]
        for patch in v['patches']:
            patch_count = patch['patch_count']
            pixel_coordinates = patch['pixel_coordinate']
            x0, y0, x1, y1 = pixel_coordinates[0][0], pixel_coordinates[0][1], pixel_coordinates[1][0], pixel_coordinates[1][1]
            area = (x1-x0)*(y1-y0)
            if area <= 1:
                print(f'{city} patch{patch_count}: {x0}, {y0}, {x1}, {y1}')
            img_p = image_full[y0:y1, x0:x1]
            patch_file_to_save = os.path.join(patch_save_dir, f'{patch_save_name}_patch{patch_count}.png')
            cv2.imwrite(patch_file_to_save, img_p)
            p_c += 1
        print(f'patch count: {p_c} for {city} {image_name} ')