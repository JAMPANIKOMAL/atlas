# v3 Environment and Installation Guide

This guide provides the official method for setting up the atlas Conda environment with full GPU support for TensorFlow.

## Step 1: Create the Environment from File

Place the `environment-v3.yml` file in your project's root directory.

Open a Conda terminal, navigate to the project root, and run the following command:

```bash
conda env create -f environment-v3.yml
```

## Step 2: Install GPU Support Libraries (CUDA & cuDNN)

After the base environment is created, you must install the specific CUDA Toolkit and cuDNN versions that are compatible with TensorFlow 2.10.

Activate the newly created environment:

```bash
conda activate atlas
```

Install the required libraries from the conda-forge channel:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## Step 3: Verify the Installation

To confirm that TensorFlow can successfully detect and use your GPU, run the following command from your activated atlas-v3 environment:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Expected Successful Output

You should see your GPU listed, for example:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If you see this output, your environment is correctly configured and ready for GPU-accelerated model training.