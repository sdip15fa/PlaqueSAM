## PlaqueSAM Demo Data Input Format

The input data for PlaqueSAM must be organized in the following folder structure:

```python
JPEGImages/
├── Patient_01/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   ├── image5.jpg
│   └── image6.jpg
├── Patient_02/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   ├── image5.jpg
│   └── image6.jpg
├── Patient_03/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   ├── image5.jpg
│   └── image6.jpg
...
```

## Directory Description
1. JPEGImages/: The root directory containing all patient image data.
2. Subfolders: Each subfolder represents a patient. The folder name can be customized (e.g., Patient_01, Patient_02, etc.).
3. Image Files: **Each patient folder must contain exactly 6 .jpg images**. The filenames can be customized but must have the .jpg extension.
4. The resolution and size of the images can vary, but they must be valid .jpg files.
5. Avoid using special characters (e.g., spaces, non-ASCII characters) in folder and file names. Use letters, numbers, and underscores instead.
