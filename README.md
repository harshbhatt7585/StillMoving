# Still-Moving: Open-Source Implementation

## About

This repository contains an open-source implementation of the "Still-Moving" model, based on the paper "Still-Moving: Customized Video Generation without Customized Video Data" by Chefer et al.
[project page](https://still-moving.github.io/)

Still-Moving is a novel framework for customizing text-to-video (T2V) generation models without requiring customized video data. It leverages customized text-to-image (T2I) models and adapts them for video generation, combining spatial priors from T2I models with motion priors from T2V models.

## Key Features

- Customization of T2V models using only still image data
- Support for personalization, stylization, and conditional generation
- Implementation of Motion Adapters and Spatial Adapters
- Compatible with different T2V architectures (e.g., Lumiere, AnimateDiff)

## Installation

[Include installation instructions here]

## Usage

[Provide basic usage examples here]

## Implementation Details

- Motion Adapters: LoRA layers applied to temporal attention blocks
- Spatial Adapters: LoRA layers added after injected customized T2I layers
- Training process: Two-step training for Motion and Spatial Adapters
- Supported models: [List the T2V models you've implemented]

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or proposing new features, your efforts are appreciated.

Please make sure to update tests as appropriate and adhere to the project's coding standards.

### Areas for Contribution

- Implementing support for additional T2V models
- Optimizing performance and reducing computational requirements
- Improving documentation and adding usage examples
- Creating tools for easier model customization
- Developing a user-friendly interface for video generation

## License

Open to use

## Contact

Harsh Bhatt - harshbhatt7585@gmail.com
