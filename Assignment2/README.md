# CIFAR-10 CNN Assignment

## Files
- `data_preperation.py`: CIFAR-10 loaders, augmentation, class grid, pixel/class analysis, one-hot helper.
- `cnn_model.py`: baseline CNN, mini-VGGNet, residual CNN, parameter summary, filter and feature-map plots.
- `eval.py`: training loop, early stopping, best checkpoint, learning-rate scheduler, curves, confusion matrix, class accuracy, misclassified examples, Grad-CAM.
- `experiment.py`: pretrained ResNet18/MobileNetV2 constructors, ensemble probability averaging, horizontal-flip test-time augmentation.
- `assignment.ipynb`: notebook walkthrough for the assignment.

## Setup
```bash
pip install torch torchvision matplotlib numpy scikit-learn jupyter
jupyter notebook assignment.ipynb
```

The notebook downloads CIFAR-10 into `./data` on first run. A CUDA-capable GPU is recommended for 50–100 epochs. Begin with 2 epochs to test the environment.

## Notes
- The implementation uses PyTorch and standardizes images with CIFAR-10 channel mean/std.
- `CrossEntropyLoss` expects integer labels and raw logits; softmax is used only for probability averaging/inference.
- The train/validation split is seeded; validation and test transforms are deterministic.
- Transfer learning requires a separate loader using `get_transfer_transforms`, because those models expect 224×224 ImageNet-normalized inputs.
- Ensemble inference assumes all models receive identically preprocessed inputs.
- No accuracy numbers are fabricated: run the notebook to obtain results.
