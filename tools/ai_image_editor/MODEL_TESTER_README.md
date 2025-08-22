# DeepFace Model Tester

This script automatically tests different DeepFace models with your test images to find the best performing model and threshold combination for face verification.

## 🚀 Quick Start

1. **Prepare your test images:**
   - Create a folder called `test_images` in the same directory as this script
   - Add 3 test images to the folder:
     - `same_person_1.jpg` - First image of the same person
     - `same_person_2.jpg` - Second image of the same person  
     - `different_person.jpg` - Image of a different person

2. **Run the tester:**
   ```bash
   python model_tester.py
   ```

3. **Review the results:**
   - The script will test all DeepFace models with different distance metrics and thresholds
   - It will show you the best performing combinations
   - Results are saved to `model_test_results.json`

## 📊 What it tests

The script tests:
- **7 DeepFace models:** VGG-Face, Facenet, Facenet512, OpenFace, DeepID, ArcFace, SFace
- **3 Distance metrics:** cosine, euclidean, euclidean_l2
- **9 Thresholds:** 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

**Total combinations tested:** 7 × 3 × 9 = 189 combinations

## 🎯 What it measures

For each combination, it tests:
1. **Same person test:** Should return `True` (same person)
2. **Different person test:** Should return `False` (different person)
3. **Accuracy:** Percentage of correct results
4. **Speed:** Average time per comparison

## 📋 Output

The script will show:
- **Perfect matches:** Combinations with 100% accuracy
- **Best overall:** Highest accuracy combination
- **Fastest perfect:** Fastest combination with 100% accuracy
- **Top 10 combinations:** Ranked by accuracy and speed

## 🔧 How to use the results

Once you find the best combination, update your `editor.py`:

```python
# Replace these lines in your verify_faces function:
MODEL_NAME = "BEST_MODEL_FROM_RESULTS"
THRESHOLD = BEST_THRESHOLD_FROM_RESULTS
distance_metric="BEST_DISTANCE_METRIC_FROM_RESULTS"
```

## 📁 File structure

```
your_project/
├── model_tester.py              # This script
├── MODEL_TESTER_README.md       # This file
├── test_images/                 # Your test images folder
│   ├── same_person_1.jpg
│   ├── same_person_2.jpg
│   └── different_person.jpg
├── model_test_results.json      # Generated results
└── editor.py                    # Your main application
```

## ⚠️ Important Notes

- Make sure your test images are clear and show faces prominently
- The script uses `enforce_detection=False` to avoid detection errors
- Test images should be in JPG format
- The script will create the `test_images` folder if it doesn't exist

## 🎯 Example Output

```
🏆 BEST OVERALL PERFORMANCE:
  Model: VGG-Face
  Distance Metric: euclidean
  Threshold: 0.3
  Accuracy: 100.0%
  Average Time: 0.245s

📝 TO UPDATE YOUR EDITOR.PY:

For Photo Upload verification, use:
MODEL_NAME = "VGG-Face"
THRESHOLD = 0.3
distance_metric="euclidean"

This combination achieved 100.0% accuracy on your test images.
```
