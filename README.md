# CUDA Median Filter

A CUDA and OpenCV experiment implementing a 3 × 3 median filter for grayscale images.

<p>
  <a href="https://buymeacoffee.com/lucky_1uck">
    <img src="https://img.shields.io/badge/Support_the_project-FFDD00?style=for-the-badge&amp;logo=buymeacoffee&amp;logoColor=black" alt="Support the project" width="260" height="40">
  </a>
</p>

## Algorithm

For each pixel, the CUDA kernel collects a 3 × 3 neighbourhood, sorts its nine
values and writes the middle value to the output image. The host code copies
image data to the GPU and reports elapsed GPU time with CUDA events.

## Source and samples

| File | Purpose |
| --- | --- |
| [kernel.cu](kernel.cu) | CUDA kernel, host-side filter and demonstration entry point |
| [image.bmp](image.bmp) | Sample input image |
| [result.bmp](result.bmp) | Checked-in sample output |

## Build requirements

- An NVIDIA GPU and a compatible CUDA driver/toolkit.
- A C++ compiler supported by that CUDA toolkit.
- OpenCV headers and libraries configured in the compiler's include/link paths.

The source uses legacy CUDA texture references (`texture<>` and
`cudaBindTexture2D`). It needs a compatible older toolkit or a texture-object
migration for toolkits that removed those APIs. No CMake or IDE build project
is included; compile `kernel.cu` with `nvcc` and your OpenCV include/link options.

## Current demo behaviour

`main()` reads `image.bmp` as grayscale and writes it to `result.bmp`.
The allocation of the filtered output and call to `medianFilter` are commented
out. Enable those lines and write the filtered matrix to exercise the GPU
filter; running the unchanged demo only copies the input image.

## ☕ Support the project

If this project helps you learn or build, you can support its development.

<p>
  <a href="https://buymeacoffee.com/lucky_1uck">
    <img src="https://img.shields.io/badge/Support_the_project-FFDD00?style=for-the-badge&amp;logo=buymeacoffee&amp;logoColor=black" alt="Support the project" width="260" height="40">
  </a>
</p>
