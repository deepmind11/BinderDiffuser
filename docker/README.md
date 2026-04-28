# Docker image

The Dockerfile bundles RFdiffusion, ProteinMPNN, ColabFold, and BinderDiffuser
into a single GPU-ready image so the pipeline runs end-to-end in one container.

## Build

```bash
docker build -t binderdiffuser:latest -f docker/Dockerfile .
```

The build pulls model weights for RFdiffusion lazily on first use; expect a
~15 GB image once weights cache.

## Run

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  binderdiffuser:latest \
  run /workspace/examples/pdl1_binder/config.yaml
```

## CPU-only fallback

ColabFold has a CPU code path but inference is ~50x slower; use it only for
development sanity checks. The recommended path on a Mac M-series is the
Colab notebook in `notebooks/03_colab_inference.ipynb`, which runs on a free
T4 GPU.

## Mounting weights

If you have RFdiffusion weights pre-downloaded on the host, mount them at
`/opt/RFdiffusion/models`:

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -v $HOME/rfdiff_weights:/opt/RFdiffusion/models \
  binderdiffuser:latest run /workspace/examples/pdl1_binder/config.yaml
```
