# 📂 Data Directory

This directory should contain the **SUN 397** scene classification dataset.

## Expected Structure

```
data/
├── images/          # 62,145 scene images (JPG/PNG)
├── TRAIN.csv        # Training set: IMAGE → LABEL mapping (37,287 samples)
└── TEST.csv         # Test set: IMAGE filenames for inference (24,858 samples)
```

## Dataset Details

| Property     | Value                            |
| ------------ | -------------------------------- |
| Total Images | 62,145                           |
| Train Split  | 37,287 images                    |
| Test Split   | 24,858 images                    |
| Classes      | 397 scene categories             |
| Source        | SUN 397 (Scene Understanding)   |
| Size          | ~22 GB                          |

## Download

The dataset is based on the [SUN 397](https://vision.princeton.edu/projects/2010/SUN/) benchmark for scene understanding. Due to its large size, it is **not included** in this repository.

After downloading, place the files in this directory and ensure the filenames match the structure above.
