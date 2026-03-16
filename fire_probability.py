import argparse
import os
import sys
import numpy as np
import rasterio
import h5py
import tensorflow as tf
import xarray as xr
from tensorflow import keras

import numba

from numba import jit
from numba import cuda
from numba import config

import subprocess

# Disable TF logging noise for cleaner terminal output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NN Prediction and Probability Pipeline"
    )

    parser.add_argument(
        "-n",
        "--nn_path",
        type=str,
        required=True,
        help="Path to the saved fuel-type classification Keras model (.keras)",
    )

    parser.add_argument(
        "-s",
        "--scene_bands",
        type=str,
        required=True,
        help="File path of Sentinel-2 scene bands (for fuel-type Prediction)",
    )
    parser.add_argument(
        "-v",
        "--ndvi",
        type=str,
        required=True,
        help="File path of NDVI scene (for Probability Calc)",
    )
    parser.add_argument(
        "-m",
        "--ndmi",
        type=str,
        required=True,
        help="File path oh NDMI scene (for Probability Calc)",
    )

    parser.add_argument(
        "-f",
        "--fuel_type_path",
        type=str,
        required=True,
        help="Path to save the fuel-type GeoTIFF",
    )

    parser.add_argument(
        "-l",
        "--likelihood_path",
        type=str,
        required=True,
        help="Path to the probability likelihood file (.h5)",
    )

    parser.add_argument(
        "-o",
        "--final_output_path",
        type=str,
        required=True,
        help="Path to save the final probability GeoTIFF",
    )

    parser.add_argument(
        "--use_gpu",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Use GPU for processing (True/False)",
    )

    return parser.parse_args()

def is_use_gpu(use_gpu):
    """
    Step 1: Check GPU availability and configure TensorFlow based on user preference.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(f"[INFO] Detected {len(gpus)} GPU(s).")
        if not use_gpu:
            print("[INFO] User requested CPU only. Hiding GPUs from all processing.")
            tf.config.set_visible_devices([], "GPU")
            return False
        else:
            print("[INFO] GPU usage enabled. Processing will utilize available GPUs.")
            return True
    else:
        print("[INFO] No GPU detected. Running on CPU.")
        if use_gpu:
            print("[WARNING] GPU requested but none available. Proceeding on CPU.")
        return False

def read_raster_all_bands(file_path):
    """
    Step 2: Read file input using rasterio (all bands).
    Returns: data (numpy array), profile (metadata), transform, crs
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with rasterio.open(file_path) as src:
        data = src.read()  # Reads all bands
        profile = src.profile
        transform = src.transform
        crs = src.crs
        img = np.ma.masked_where(data == 1e20, data)
        img = np.ma.masked_invalid(img)

    return data, img, profile, transform, crs

def save_geotiff(data, profile, transform, crs, output_path):
    """
    Helper to save numpy array as GeoTIFF
    """
    out_profile = profile.copy()

    # Handle data dimensions for rasterio (Bands, Height, Width)
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)  # Add band dimension if missing

    out_profile.update(
        {
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "count": data.shape[0],
            "dtype": data.dtype,
            "transform": transform,
            "crs": crs,
        }
    )

    # Ensure directory exists
    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)), exist_ok=True
    ) if os.path.dirname(output_path) else None

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(data)
    print(f"[INFO] Saved GeoTIFF to: {output_path}")

def load_probability_likelihood(likelihood_path):
    """
    Load parameters from HDF5 file.
    Adjust this function based on the actual structure of your HDF5 file.
    """
    if not os.path.exists(likelihood_path):
        raise FileNotFoundError(f"HDF5 file not found: {likelihood_path}")

    prob_table = xr.open_dataarray(likelihood_path)
    return prob_table

def classify_fuel_type(data_images, model):
    """
    Doing fuel type classification using saved Keras Model
    """
    fuel_type_model = keras.Sequential([model, keras.layers.Softmax()])

    prediction_dataset = np.zeros((data_images.shape[1] * data_images.shape[2], data_images.shape[0]))

    # Reshape the raster file shape from (Bands, Y, X) to (Y * X, Bands), so it conformant to the input layer of Neural Network.
    for y in range(data_images.shape[1]):
        for x in range(data_images.shape[2]):
            count = (y * data_images.shape[1]) + x
            prediction_dataset[count] = data_images[:, y, x]

    # Create input tensor from the reshaped raster file
    prediction_tensor = tf.data.Dataset.from_tensor_slices((prediction_dataset))
    prediction_tensor = prediction_tensor.batch(64)
    prediction_tensor = prediction_tensor.prefetch(64)

    # Doing prediction
    prediction = fuel_type_model.predict(prediction_tensor)

    # Based on softmax class probability, decide the class using argmax
    prediction_result = np.zeros(prediction.shape[0])
    for i, p in enumerate(prediction):
        pred = np.argmax(p)
        prediction_result[i] = pred

    # Reshape, then cloud mask
    pred_final = prediction_result.reshape((data_images.shape[1], data_images.shape[2]))
    cloud_mask = np.ma.masked_invalid(data_images[0])
    pred_final = np.ma.masked_array(pred_final, cloud_mask.mask)

    return pred_final

def pixel_processing(indexes, likelihood_prob_table, prior, result):
  index_width = indexes.sizes['x']
  index_height = indexes.sizes['y']
  for y in range(index_height):
      for x in range(index_width):
          pixel_vals = indexes.isel(y=y, x=x)
          ndvi_pixel = pixel_vals.sel(index='ndvi').values.round(1)
          ndmi_pixel = pixel_vals.sel(index='ndmi').values.round(1)
          fuel_type_pixel = pixel_vals.sel(index='fuel_type').values.round(1)

          print(x, y)
          print(ndvi_pixel, ndmi_pixel, "\n")
          if np.isnan(ndvi_pixel) or np.isnan(ndmi_pixel) or np.isnan(fuel_type_pixel):
              result[y, x] = np.nan
              continue

          try:
              likelihood = likelihood_prob_table.sel(burned=True, ndvi=ndvi_pixel, ndmi=ndmi_pixel, fuel_type=fuel_type_pixel).values
              result[y, x] = likelihood
          except KeyError:
              result[y, x] = np.nan

@jit
def pixel_processing_cuda(image_array, likelihood_table, dims, burned_vals, ndvi_vals, ndmi_vals, fuel_type_vals, result):
  for j in range(result.shape[0]):
    for i in range(result.shape[1]):
      # image_index = j * result.shape[1] + i
      tmp = 0.0
      for val_idx in range(image_array.shape[0]):
        if dims[val_idx + 1] == 'ndvi':
          ndvi_coord = 0
          for ids, da in enumerate(ndvi_vals):
            if da == image_array[val_idx][j][i]:
              ndvi_coord = ids
              break

        elif dims[val_idx + 1] == 'ndmi':
          ndmi_coord = 0
          for ids, da in enumerate(ndmi_vals):
            if da == image_array[val_idx][j][i]:
              ndmi_coord = ids
              break

        elif dims[val_idx + 1] == 'fuel_type':
          fuel_type_coord = 0
          for ids, da in enumerate(fuel_type_vals):
            if da == image_array[val_idx][j][i]:
              fuel_type_coord = ids
              break

      result[j][i] = likelihood_table[0][ndvi_coord][ndmi_coord][fuel_type_coord]

  return result

def bayes_predict(prob_table, indicies, prior, use_gpu):
  indexes = indicies
  index_width = indexes.sizes['x']
  index_height = indexes.sizes['y']

  post_img = np.zeros((index_height, index_width))

  # flatten_indexes = np.array([ indexes.values[:, y, x] for y in range(post_img.shape[0]) for x in range(post_img.shape[1])]).round(decimals=1)
  # likelihood_prob_table = xr.open_dataarray(HOME_DIR + "generated_data/bayesian_likelihood.h5")

  if not use_gpu:
    pixel_processing(indexes, prob_table, prior, post_img)
  else:
    image_array = indicies.values
    dims = prob_table.dims
    burned_vals = prob_table.burned.values
    ndvi_vals = prob_table.ndvi.values
    ndmi_vals = prob_table.ndmi.values
    fuel_type_vals = prob_table.fuel_type.values
    likelihood_table_vals = prob_table.values
    post_img = np.zeros_like(image_array[0])

    pixel_processing_cuda(image_array, likelihood_table_vals, dims, burned_vals, ndvi_vals, ndmi_vals, fuel_type_vals, post_img)

  posterior_img = post_img * prior / ((post_img * prior) + ((1 - post_img) * (1 - prior)))

  return posterior_img

def main():
    args = parse_arguments()

    # 1. Check GPU availability and configure
    use_gpu = is_use_gpu(args.use_gpu)

    try:
        # Read Sentinel-2 scene bands.
        print("[INFO] Reading Sentinel-2 scene bands")
        scene_bands, masked_sb, profile_sb, transform_sb, crs_sb = read_raster_all_bands(args.scene_bands)

        # Neural-Network Prediction (Model + Scene bands)
        print("[INFO] Loading Keras Model...")
        model = keras.models.load_model(args.nn_path)
        fuel_type_array = classify_fuel_type(scene_bands, model)

        save_geotiff(fuel_type_array, profile_sb, transform_sb, crs_sb, args.fuel_type_path)

        # Read NDVI index
        print("[INFO] Reading NDVI index...")
        ndvi, masked_ndvi, profile_ndvi, transform_ndvi, crs_ndvi = read_raster_all_bands(args.ndvi)

        # Read NDMI index
        print("[INFO] Reading NDMI index...")
        ndmi, masked_ndmi, profile_ndmi, transform_ndmi, crs_ndmi = read_raster_all_bands(args.ndmi)

        # Read Fuel Type
        print("[INFO] Reading fuel type classes...")
        fuel_type, masked_fuel_type, profile_fuel_type, transform_fuel_type, crs_fuel_type = read_raster_all_bands(args.fuel_type_path)

        # Load HDF5
        print("[INFO] Loading HDF5 parameters...")
        likelihood_prob_table = load_probability_likelihood(args.likelihood_path)

        print("[INFO] Using pre-defined prior: 0.5919471011925848")
        prior = 0.5919471011925848 # based on learning on bromo 2023 data
        
        ndvi_ind = np.round(ndvi[0, :1000, :1000], decimals=1)
        ndmi_ind = np.round(ndmi[0, :1000, :1000], decimals=1)
        fuel_type_ind = np.round(masked_fuel_type[0, :1000, :1000], decimals=1)

        data_indicies = xr.DataArray(np.array([ndvi_ind, ndmi_ind, fuel_type_ind]), dims=["index", "y", "x"], coords={"index": ["ndvi", "ndmi", "fuel_type"]})

        predicted_fire_prob = bayes_predict(likelihood_prob_table, data_indicies, prior, use_gpu)

        print("[SUCCESS] Pipeline completed successfully.")

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
