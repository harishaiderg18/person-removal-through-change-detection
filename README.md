# Person Removal from Video Frames using Background Subtraction

A complete computer vision pipeline for detecting and removing moving persons from video sequences using background subtraction, morphological operations, and alpha blending techniques. Implements all algorithms from scratch using only standard libraries (numpy, matplotlib, pathlib, struct, zlib, math, collections, and cv2.VideoWriter).

---

## Project Structure

```
computer vision/
├── main.ipynb          # Main Jupyter notebook with all implementations
├── README.md                          # This file
├── Results/
│   ├── person1_masks.npy             # Saved binary masks for person1
│   ├── person3_masks.npy             # Saved binary masks for person3
│   └── person3_Output/               # Output frames
├── Assignment1_Outputperson3/        # Additional outputs
└── person_removal_images/            # Generated images
```

---

##  Overview

The notebook contains a complete implementation that:
1. Builds a statistical background model (mean and variance)
2. Detects foreground objects using Mahalanobis distance
3. Cleans binary masks with custom morphological operations
4. Removes persons via progressive alpha blending
5. Generates visualization reports and output videos

---

##  Requirements

### Python Dependencies
```bash
pip install numpy matplotlib opencv-python
```

- `numpy` - Numerical computations
- `matplotlib` - Visualization and plotting
- `opencv-python` (cv2) - Video writing only
- `pathlib` - File path handling
- `struct`, `zlib` - Custom PNG file writing

---

##  Key Functions

### 1. Frame Reading and Preprocessing
- **`read_frames(input_folder, limit=None)`**: Loads image frames from folder, converts to grayscale, returns numpy array

### 2. Background Modeling
- **`compute_mean(frames)`**: Computes pixel-wise mean across frames
- **`compute_variance(frames, mean)`**: Computes pixel-wise variance
- **`build_background_model(frames, t)`**: Builds background model from first t frames

### 3. Foreground Detection
- **`compute_mask(frame, mean_frame, variance_frame, threshold=5.0)`**: Detects foreground using Mahalanobis distance thresholding

### 4. Morphological Operations (Custom Implementation)
- **`create_kernel(kernel_size=3)`**: Creates square structuring element
- **`erode(mask, kernel, anchor=None, iterations=1)`**: Custom erosion (no OpenCV)
- **`dilate(mask, kernel, anchor=None, iterations=1)`**: Custom dilation (no OpenCV)
- **`morphological_operations(mask, kernel_size=3)`**: Applies opening (erosion + dilation)

### 5. Connected Components Analysis
- **`find_connected_components(mask, connectivity=8)`**: Labels connected regions using BFS algorithm

### 6. Person Removal
- **`clean_all_masks(masks, kernel_size=3)`**: Applies morphological cleaning to all masks
- **`remove_person_alpha_blending(frames_with_person, background_frame, masks)`**: Removes person using progressive alpha blending

### 7. Output Generation
- **`write_png(img, filename)`**: Custom PNG writer (no external libraries)
- **`save_alphavideo(frames, out_dir, video_name, fps=10)`**: Saves frames as MP4 video
- **`make_pdf_report(...)`**: Creates multi-page PDF report
- **`plot_frames(frames, num_frames, save_name)`**: Plots multiple frames in grid

---

##  Usage

### Running the Notebook

1. **Open notebook:**
   ```
   main.ipynb
   ```

2. **Configure input paths in Cell 2:**
   ```python
   input_folder3 = r"C:\Users\user\Downloads\Augmented_person"
   input_folder1 = r"C:\Users\user\Downloads\person1"
   ```

3. **Run cells sequentially** or execute the complete pipeline:
   ```python
   if __name__ == "__main__":
       main()
   ```

### Basic Workflow Example

```python
# 1. Load frames
frames = read_frames(input_folder)

# 2. Build background model (first 70 frames)
mean_img, var_img = build_background_model(frames, 70)

# 3. Generate foreground masks
masks = generate_and_save_masks(frames, mean_img, var_img, 
                                threshold=2.0, 
                                save_path="masks.npy")

# 4. Clean masks with morphology
cleaned_masks = clean_all_masks(masks, kernel_size=3)

# 5. Remove person using alpha blending
background = mean_img.astype(np.uint8)
new_frames = remove_person_alpha_blending(frames, background, cleaned_masks)

# 6. Save results
make_pdf_report(frames, cleaned_masks, new_frames, "report.pdf")
save_alphavideo(new_frames, output_dir, "output_video.mp4", fps=10)
```

---

##  Output Files

| File | Description |
|------|-------------|
| `p1mean.png` / `p3mean.png` | Background mean images |
| `p1variance.png` / `p3variance.png` | Normalized variance images |
| `person1_masks.npy` / `person3_masks.npy` | Binary foreground masks (numpy array) |
| `person_removal_report.pdf` | Multi-page PDF showing original, mask, and result for each frame |
| `alpha_blend_video.mp4` | Final video with person removed |

---

##  Algorithm Details

### Background Subtraction
Uses Gaussian background model where each pixel is modeled by mean (μ) and variance (σ²). Foreground detection uses Mahalanobis distance:

$$d^2 = \frac{(I(x,y) - \mu(x,y))^2}{\sigma^2(x,y) + \epsilon}$$

Pixels with $d^2 > \text{threshold}$ are classified as foreground.

### Morphological Cleaning
- **Opening** operation (erosion followed by dilation) removes noise and small artifacts
- Custom implementations without using OpenCV morphology functions
- Uses square structuring elements

### Progressive Alpha Blending
Gradual person removal where alpha increases linearly across frames:

$$I_{\text{new}}(i) = I_{\text{orig}}(i) \cdot (1 - \alpha_i \cdot M(i)) + I_{\text{bg}} \cdot (\alpha_i \cdot M(i))$$

Where:
- $M(i)$ is the binary mask for frame $i$
- $\alpha_i = \min(\frac{i+1}{N}, 1.0)$ increases over time
- $I_{\text{bg}}$ is the background model

---

##  Parameters to Tune

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Background frames (t)** | 70 (person1), 60 (person3) | Number of initial frames for background model |
| **Threshold** | 2.0 | Mahalanobis distance threshold for foreground detection |
| **Kernel size** | 3 | Structuring element size for morphology |
| **FPS** | 10 | Output video frame rate |

---

##  Input Requirements

- **Format**: Directory containing sequential image frames (PNG, JPG, JPEG)
- **Image Type**: Color or grayscale (automatically converted to grayscale)
- **Naming**: Files should be sortable alphabetically/numerically
- **Background Assumption**: First N frames should be background-only (or mostly background)

---

##  Key Features

✅ Custom PNG writer (no external image libraries needed)  
✅ From-scratch morphological operations (no OpenCV morphology)  
✅ Connected components with BFS (8-connectivity)  
✅ Progressive alpha blending for smooth temporal transitions  
✅ Comprehensive visualization tools (PDF reports, video output)  
✅ Efficient numpy-based implementations  

---

## 🎓 Assignment Context

**Computer Vision - Assignment 1: Person Removal from Video Sequences**

This implementation demonstrates classical computer vision techniques for:
- Statistical background modeling
- Change detection
- Binary image processing
- Object segmentation
- Video composition

---

##  Notes

- All core algorithms implemented from scratch (except video writing)
- No OpenCV used for morphological operations or connected components
- Supports both person1 and person3 datasets with different parameters
- Output locations default to Desktop for easy access
