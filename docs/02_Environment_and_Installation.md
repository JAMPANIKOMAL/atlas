# ATLAS Environment and Installation Guide

This guide provides instructions for setting up a Conda environment to run the ATLAS pipelines. The project supports two distinct workflows: a GPU-accelerated setup for users with compatible NVIDIA hardware, and a CPU-only setup for all other users.

Please choose the guide that matches your system.

## Option 1: GPU-Accelerated Environment (Recommended for Performance)

This setup is for users with an NVIDIA GPU and provides the fastest performance for model training. The environment is configured to use TensorFlow 2.10, which requires a specific version of the CUDA Toolkit and cuDNN.

### Step 1: Create the Base Environment

You will need the `environment.yml` file located in the project's root directory.

Open a Conda terminal, navigate to the project root, and run the following command:

```bash
conda env create -f environment.yml
```

This will create an environment named `atlas` with all the necessary Python packages.

### Step 2: Install GPU Support Libraries

After the base environment is created, you must install the specific CUDA Toolkit and cuDNN versions.

Activate the newly created environment:

```bash
conda activate atlas
```

Install the required libraries from the conda-forge channel:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Step 3: Verify the GPU Installation

To confirm that TensorFlow can successfully detect and use your GPU, run the following command from your activated atlas environment:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected Successful Output:**
You should see your GPU listed, for example: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

If you see this output, your environment is correctly configured for GPU-accelerated training.

## Option 2: CPU-Only Environment (Universal Compatibility)

This setup is for users without an NVIDIA GPU or those who prefer a simpler installation process. The CPU-only environment uses TensorFlow-CPU and provides good performance for inference and smaller-scale training tasks.

### Step 1: Create the CPU Environment

You will need the `environment-cpu.yml` file located in the project's root directory.

Open a Conda terminal, navigate to the project root, and run the following command:

```bash
conda env create -f environment-cpu.yml
```

This will create an environment named `atlas-cpu` with all the necessary Python packages optimized for CPU usage.

### Step 2: Activate the Environment

Activate the newly created environment:

```bash
conda activate atlas-cpu
```

### Step 3: Verify the CPU Installation

To confirm that the environment is working correctly, run the following command:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version {tf.__version__} is installed.')"
```

**Expected Successful Output:**
You should see a message confirming the TensorFlow version, for example: `TensorFlow version 2.10.0 is installed.`

Your CPU-only environment is now ready and can be used to run all ATLAS pipelines.