from sentinelhub import SHConfig
import sentinelhub
import datetime
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import requests
from typing import Any
import imageio


from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

def get_token():
    """
    Gets a token used to make sentinel api requests
    oauth info loaded from the config file, which is already updated
    with oauth info
    """
    config = SHConfig()
    client_id = config.sh_client_id
    client_secret = config.sh_client_secret
    url = config.sh_token_url

    payload = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret
    }

    # Make the POST request to get the access token
    response = requests.post(url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        print(f"Access token created")
        return access_token
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def bbox_square(lon, lat, length, precision=8):
    """
    Accepts longitude and latitude of lower left bbox corner
    Accepts length of bbox in km
    Accepts number of decimal places as precision, default to 8
    Returns a bounding box object
    """
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError((f"Invalid coordinates: Latitude should be between -90 and 90, and Longitude should be between -180 and 180. Given lat: {lat}, lon: {lon}"))

    # calculate latitude change
    lat_change = length / 111.0

    # Calculate the longitude change, accounting for the latitude
    lon_change = length / (111.0 * math.cos(math.radians(lat)))

    # Calculate the upper right corner coordinates
    upper_right_lon = lon + lon_change
    upper_right_lat = lat + lat_change

    lon = round(lon, precision)
    lat = round(lat, precision)
    upper_right_lon = round(upper_right_lon, precision)
    upper_right_lat = round(upper_right_lat, precision)

    bbox_coordinates = (lon, lat, upper_right_lon, upper_right_lat)
    return BBox(bbox=bbox_coordinates, crs=CRS.WGS84)

def calc_bbox_size(bbox, resolution):
    return bbox_to_dimensions(bbox, resolution=resolution)

def get_image_square(lon, lat, length):
    """ 
    Accepts longitude in float form
    Accepts latitude in float form
    Accelts side length (in km) in float form
    returns satellite image of box with side length length, and upper left corner of image at the passed coordinates
    """
    config = sentinelhub.SHConfig()
    bbox = bbox_square(lon, lat, length)
    bbox_size = calc_bbox_size(bbox, 10)
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=("2023-08-01", "2023-09-15"),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    return image

def canvas_region_rectangular(lon_lower_left, lat_lower_left, lon_upper_right, lat_upper_right, tile_size=2):
    """
    Accepts lon and lat for the lower left corner of the desired region.
    Accepts lon and lat for upper right corner of the desired region
    Accepts a tile size for each image (in km)
    Returns a list of bounding box objects (square)
    """

    def get_bbox_row(ll_lon, ll_lat, ur_lon):
        last_bbox = bbox_square(ll_lon, ll_lat, tile_size)
        row = [last_bbox]

        while last_bbox.upper_right[0] < ur_lon:
            last_bbox = row[-1]
            # set bbox lower left lon equal to lon on last bbox right side
            bbox_ll_lon = last_bbox.upper_right[0]

            # keep bbox lower left lat consistent across the whole row
            bbox_ll_lat = last_bbox.lower_left[1]

            # create and append a new bounding box to the row
            new_bbox = bbox_square(bbox_ll_lon, bbox_ll_lat, tile_size)
            row.append(new_bbox)
        
        # add one final bbox to row to ensure complete coverate of the provided coordinates
        last_bbox = row[-1]
        bbox_ll_lon = last_bbox.upper_right[0]
        bbox_ll_lat = last_bbox.lower_left[1]
        new_bbox = bbox_square(bbox_ll_lon, bbox_ll_lat, tile_size)
        row.append(new_bbox)

        return row


    first_row = get_bbox_row(lon_lower_left, lat_lower_left, lon_upper_right)
    bbox_grid = [first_row]
    last_lat = lat_lower_left


    while last_lat < lat_upper_right:

        # for each new row, get the lower left and upper right bbox coordinates for the first element of the prior row
        last_lower_left_row_start = bbox_grid[-1][0].lower_left
        last_upper_right_row_start = bbox_grid[-1][0].upper_right

        # set a new lower left bbox coordinate for a new row using the prior lower left lon and upper right lat
        new_lower_left_lon = last_lower_left_row_start[0]
        new_lower_left_lat = last_upper_right_row_start[1]

        # get the next row of the area and append it to the row
        next_row = get_bbox_row(new_lower_left_lon, new_lower_left_lat, lon_upper_right)
        bbox_grid.append(next_row)

        last_lat = new_lower_left_lat

    # add the final row to grid to ensure complete coverate of provided area
    last_lower_left_row_start = bbox_grid[-1][0].lower_left
    last_upper_right_row_start = bbox_grid[-1][0].upper_right
    new_lower_left_lon = last_lower_left_row_start[0]
    new_lower_left_lat = last_upper_right_row_start[1]
    next_row = get_bbox_row(new_lower_left_lon, new_lower_left_lat, lon_upper_right)
    bbox_grid.append(next_row)

    return bbox_grid



def get_images_from_region(region_lon_lower_left, region_lat_lower_left, region_lon_upper_right, region_lat_upper_right, output_dir, tile_size=2):
    bounding_box_array_2d = canvas_region_rectangular(region_lon_lower_left, region_lat_lower_left, region_lon_upper_right, region_lat_upper_right, tile_size=2)
    total_images = len(bounding_box_array_2d) * len(bounding_box_array_2d[0])
    processed = 0
    for row in range(len(bounding_box_array_2d)):
        for col in range(len(bounding_box_array_2d[row])):
            bbox = bounding_box_array_2d[row][col]
            file_name = f'{bbox.lower_left[0]},{bbox.lower_left[1]}'

            filepath = os.path.join(output_dir, file_name)
            if not os.path.exists(filepath + '.png'):
                img = get_image_square(bbox.lower_left[0], bbox.lower_left[1], tile_size)
                save_image(img, filepath=filepath + '.png', factor=5 / 255, clip_range=(0, 1))
            processed += 1
            print(f'Processed {processed} / {total_images}')


def save_image(image: np.ndarray, filepath: str, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any) -> None:
    
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)

    ax.axis('off')

    try:
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    except Exception as e:
        print(f'Error: {e}')


# bounds for olympic mountains
#upper_right =  (-123.021647, 47.973022)
#lower_left = (-123.819648, 47.382983)

# bounds for mt. adams
#upper_right =  [-121.423597, 46.266517]
#lower_left = [-121.593377, 46.134413]

# bounds for goat rocks
upper_right = (-121.393653, 46.581592)
lower_left = (-121.534044, 46.441355)


# bounds for vancouver island training set
upper_right = [-125.349722, 49.826193]
lower_left = (-125.998912, 49.383343)


# bounds for north cascades
filepath='/Users/jacksonhayward/Desktop/vancouver_island_train_set/'
#get_images_from_region(lower_left[0], lower_left[1], upper_right[0], upper_right[1], filepath, tile_size=2)


p = '/Users/jacksonhayward/Desktop/miniset'


"""
files = [img for img in os.listdir(p) if img.endswith('.png')]
for i in range(len(files)):
    file = files[i]
    new_dir = os.path.join(p, f'image_{i}')
    os.mkdir(new_dir)
    old_p = os.path.join(p, file)
    new_p = os.path.join(new_dir, file)
    os.rename(old_p, new_p)

"""