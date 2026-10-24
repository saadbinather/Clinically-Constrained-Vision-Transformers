# JHU Starter Task

This folder contains a professor-provided starter task related to 3-D medical-image segmentation.

## What the notebook does

`post_processing.ipynb`:

- installs MONAI and supporting packages;
- downloads AbdomenAtlasDemo;
- downloads SuPreM and a pretrained Swin UNETR checkpoint;
- applies compatibility fixes; and
- runs inference for vertebrae segmentation in abdominal CT scans.

This is a starter/research exercise. It does not claim an internship appointment, collaboration, or endorsement.

## Status

The notebook contains successful inference output as well as repeated setup attempts and environment errors. Re-running it requires external data, the pretrained checkpoint, a compatible GPU environment, and updated paths.
