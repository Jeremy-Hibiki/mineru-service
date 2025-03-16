# An HTTP service for MinerU[^mineru] with LitServe[^litserve]

Based on the official example [web_api](https://github.com/opendatalab/MinerU/tree/ecdd162f11f92de585cd3d921d852df895d40ca6/projects/web_api) and [multi_gpu](https://github.com/opendatalab/MinerU/tree/ecdd162f11f92de585cd3d921d852df895d40ca6/projects/multi_gpu), built into `-lite` and `-full` dockers, with several modifications:

- Docker images:The `-lite` docker image only contains the service deps and code, and the `-full` docker image also includes all the model weights needed.
- Updated model download script: `paddleocr` models are also pre-downloaded, so the service can be started in complete air-gapped environment.
- Pin PyTorch version to 2.4.1+cu124, to make `paddlepaddle-gpu` dependencies happy.
  - **Note**: The official torch dependency of `MinerU` is pinned to `>=2.2.2,<=2.3.1`, but the lowest version support CUDA 12.4 is 2.4.0, and `paddlepaddle-ocr` only support 12.3
- Output directory is mounted as static resource, and the outout is also packed into a tarball, to make it easier to get the result from the client.

[^mineru]: https://github.com/opendatalab/MinerU
[^litserve]: https://github.com/Lightning-AI/LitServe
