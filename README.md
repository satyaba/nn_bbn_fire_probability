# Fire Probability Prediction Pipeline

A minimal implementation, Python-based pipeline for wildfire probability mapping using Sentinel-2 satellite imagery, deep learning-based fuel type classification, and Bayesian probability modeling. The repository doesn't include any training step and intended to use for forward prediction. The minimum required dataset and model to run the algorithm also available in this repository.

## Overview

This pipeline combines:
- **Neural Network Classification**: Predicts fuel types from Sentinel-2 multispectral bands
- **Bayesian Probability Modeling**: Calculates fire probability using NDVI, NDMI, and fuel type indices
- **GPU Acceleration**: Supports both CPU and GPU (CUDA) processing

## Features

- ✅ Fuel type classification using Keras/TensorFlow models
- ✅ Pixel-level fire probability calculation
- ✅ GPU acceleration with CUDA (optional)
- ✅ GeoTIFF output with preserved spatial metadata
- ✅ HDF5-based probability lookup tables

---

## Requirements

### Python Environment

```bash
Python >= 3.8
```

### GPU Support (Optional but highly recommended)

For GPU acceleration, you need:
- CUDA-compatible GPU
- CUDA Toolkit version 12.8

---

## Google Colab version

To simplify usage of the project, we provide implementation in Google Colab, please visit at :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_s_7auKG4NgQv73QVRzqmxxVjWtNiQb-?usp=sharing)

---

## Installation

1. **Clone or download the script**:
```bash
git clone https://github.com/satyaba/nn_bbn_fire_probability.git ./fire-prediction
cd fire-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify GPU availability** (optional):
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Usage

### Basic Command Structure

```bash
python fire_probability.py \
  --nn_path <MODEL_PATH> \
  --scene_bands <SENTINEL2_BANDS> \
  --ndvi <NDVI_FILE> \
  --ndmi <NDMI_FILE> \
  --fuel_type_path <FUEL_TYPE_OUTPUT> \
  --likelihood_path <LIKELIHOOD_TABLE> \
  --final_output_path <PROBABILITY_OUTPUT> \
  --use_gpu True
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--nn_path` | `-n` | ✅ | Path to trained Keras model (.keras) for fuel type classification |
| `--scene_bands` | `-s` | ✅ | Sentinel-2 multispectral selected 8-bands (GeoTIFF) |
| `--ndvi` | `-v` | ✅ | NDVI raster file (GeoTIFF) |
| `--ndmi` | `-m` | ✅ | NDMI raster file (GeoTIFF) |
| `--fuel_type_path` | `-f` | ✅ | Output path for fuel type classification (GeoTIFF) |
| `--likelihood_path` | `-l` | ✅ | HDF5 file containing Bayesian likelihood table |
| `--final_output_path` | `-o` | ✅ | Output path for final fire probability map (GeoTIFF) |
| `--use_gpu` | - | ❌ | Use GPU for processing (default: True) |

---

## Example Usage

### 1. Full Pipeline with GPU

```bash
python fire_probability.py \
  --nn_path models/fuel_classifier.keras \
  --scene_bands data/sentinel2_bromo_08_2023.tif \
  --ndvi data/ndvi_bromo_08_2023.tif \
  --ndmi data/ndmi_bromo_08_2023.tif \
  --fuel_type_path output/fuel_type_classified.tif \
  --likelihood_path models/bayesian_likelihood.h5 \
  --final_output_path output/fire_probability_map.tif \
  --use_gpu True
```

### 2. CPU-Only Processing

```bash
python fire_probability.py \
  --nn_path models/fuel_classifier.keras \
  --scene_bands data/sentinel2_bromo_08_2023.tif \
  --ndvi data/ndvi_bromo_08_2023.tif \
  --ndmi data/ndmi_bromo_08_2023.tif \
  --fuel_type_path output/fuel_type_classified.tif \
  --likelihood_path models/bayesian_likelihood.h5 \
  --final_output_path output/fire_probability_map.tif \
  --use_gpu False
```

### 3. Using Short Argument Names

```bash
python fire_probability.py \
  -n models/fuel_classifier.keras \
  -s data/sentinel2_bromo_08_2023.tif \
  -v data/ndvi_bromo_08_2023.tif \
  -m data/ndmi_bromo_08_2023.tif \
  -f output/fuel_type_classified.tif \
  -l models/bayesian_likelihood.h5 \
  -o output/fire_probability_map.tif
```

---

## Input File Specifications

### 1. Sentinel-2 Scene Bands (`--scene_bands`)
- **Format**: GeoTIFF (multi-band)
- **Bands**: B02, B03, B04, B05, B06, B08, B8A, B11
- **Projection**: Should match NDVI/NDMI
- **No-data handling**: Values of `1e20` or invalid are masked

### 2. NDVI File (`--ndvi`)
- **Format**: GeoTIFF (single band)
- **Range**: Typically -1.0 to 1.0
- **Precision**: Rounded to 1 decimal place internally

### 3. NDMI File (`--ndmi`)
- **Format**: GeoTIFF (single band)
- **Range**: Typically -1.0 to 1.0
- **Precision**: Rounded to 1 decimal place internally

### 4. Neural Network Model (`--nn_path`)
- **Format**: Keras model file (.keras or .h5)
- **Output**: Logits for fuel type classes
- **Note**: Softmax is applied automatically

### 5. Likelihood Table (`--likelihood_path`)
- **Format**: HDF5 (.h5) containing xarray DataArray
- **Dimensions**: `[burned, ndvi, ndmi, fuel_type]`
- **Content**: Conditional probability P(Evidence|Burned)

---

## Output Files

### 1. Fuel Type Classification (`--fuel_type_path`)
- **Format**: GeoTIFF (single band)
- **Values**: Integer class labels (0, 1, 2, ...)
- **Spatial reference**: Inherited from input scene

### 2. Fire Probability Map (`--final_output_path`)
- **Format**: GeoTIFF (single band)
- **Values**: Float [0.0 - 1.0] representing fire probability
- **Calculation**: Bayesian posterior probability
- **Formula**: 
  ```
  P(Burned|Evidence) = [P(Evidence|Burned) × Prior] / 
                        [P(Evidence|Burned) × Prior + P(Evidence|¬Burned) × (1 - Prior)]
  ```

---

## Pipeline Workflow

```
1. Load Sentinel-2 Bands
         ↓
2. Neural Network Inference → Fuel Type Classification
         ↓
3. Save Fuel Type GeoTIFF
         ↓
4. Load NDVI, NDMI, Fuel Type
         ↓
5. Load Bayesian Likelihood Table (HDF5)
         ↓
6. Bayesian Probability Calculation (CPU/GPU)
         ↓
7. Save Fire Probability GeoTIFF
```

---

## Performance Notes

### GPU vs CPU
- **GPU**: Recommended for large scenes (>10,000 × 10,000 pixels)
- **CPU**: Sufficient for small to medium scenes
- **Current implementation**: Processes first 1000×1000 pixels (line 237-239)

### Memory Considerations
- Large rasters are loaded entirely into memory
- For very large scenes, consider tiling/chunking

---

## Troubleshooting

### Issue: "No GPU detected"
**Solution**: 
- Verify CUDA installation: `nvidia-smi`
- Check TensorFlow GPU support: 
  ```python
  import tensorflow as tf
  print(tf.test.is_built_with_cuda())
  ```
- Use `--use_gpu False` to run on CPU

### Issue: "FileNotFoundError"
**Solution**: 
- Check all input paths are correct
- Ensure files exist and are readable
- Use absolute paths if relative paths fail

### Issue: "CUDA out of memory"
**Solution**:
- Reduce processing extent (currently hardcoded to 1000×1000)
- Use CPU mode: `--use_gpu False`
- Process scene in tiles

---

## Code Modifications

### Change Processing Extent
Currently hardcoded (lines 237-239):
```python
ndvi_ind = np.round(ndvi[0, :1000, :1000], decimals=1)
ndmi_ind = np.round(ndmi[0, :1000, :1000], decimals=1)
fuel_type_ind = np.round(masked_fuel_type[0, :1000, :1000], decimals=1)
```

To process full scene:
```python
ndvi_ind = np.round(ndvi[0, :, :], decimals=1)
ndmi_ind = np.round(ndmi[0, :, :], decimals=1)
fuel_type_ind = np.round(masked_fuel_type[0, :, :], decimals=1)
```

### Modify Prior Probability
Default prior (line 234):
```python
prior = 0.5919471011925848  # based on Mt. Bromo 2023 data
```

Change based on your study area's historical fire frequency.

<!-- ---

## Citation

If you use this pipeline, please cite:

```bibtex
@article{your_paper,
  title={Multi-factor Bayesian Wildfire Probability Mapping Using Sentinel-2 and Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

---

## License

[Specify your license here]

---

## Contact

For questions or issues, contact: [your.email@domain.com]

--- -->

## Changelog

### Version 1.0
- Initial release
- Fuel type classification via Keras
- Bayesian probability calculation
- GPU acceleration support
