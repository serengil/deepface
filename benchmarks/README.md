# Benchmarks - [`Video Tutorial`](https://youtu.be/eKOZawGR3y0)

DeepFace offers various configurations that significantly impact accuracy, including the facial recognition model, face detector model, distance metric, and alignment mode. Our experiments conducted on the [LFW dataset](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/) using different combinations of these configurations yield the following results.

You can reproduce the results by executing the `Perform-Experiments.ipynb` and `Evaluate-Results.ipynb` notebooks, respectively.

## ROC Curves

ROC curves provide a valuable means of evaluating the performance of different models on a broader scale. The following illusration shows ROC curves for different facial recognition models alongside their optimal configurations yielding the highest accuracy scores.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/benchmarks.jpg" width="95%" height="95%"></p>

In summary, FaceNet-512d surpasses human-level accuracy, while FaceNet-128d reaches it, with Dlib, VGG-Face, and ArcFace closely trailing but slightly below, and GhostFaceNet and SFace making notable contributions despite not leading, while OpenFace, DeepFace, and DeepId exhibit lower performance.

## Accuracy Scores

Please note that humans achieve a 97.5% accuracy score on the same dataset. Configurations that outperform this benchmark are highlighted in bold.

## Performance Matrix for euclidean while alignment is True

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |95.9 |93.5 |95.8 |85.2 |88.9 |85.9 |80.2 |69.4 |67.0 |65.6 |
| mtcnn |95.2 |93.8 |95.9 |83.7 |89.4 |83.0 |77.4 |70.2 |66.5 |63.3 |
| fastmtcnn |96.0 |93.4 |95.8 |83.5 |91.1 |82.8 |77.7 |69.4 |66.7 |64.0 |
| dlib |96.0 |90.8 |94.5 |88.6 |96.8 |65.7 |66.3 |75.8 |63.4 |60.4 |
| yolov8 |94.4 |91.9 |95.0 |84.1 |89.2 |77.6 |73.4 |68.7 |69.0 |66.5 |
| yunet |97.3 |96.1 |96.0 |84.9 |92.2 |84.0 |79.4 |70.9 |65.8 |65.2 |
| centerface |**97.6** |95.8 |95.7 |83.6 |90.4 |82.8 |77.4 |68.9 |65.5 |62.8 |
| mediapipe |95.1 |88.6 |92.9 |73.2 |93.1 |63.2 |72.5 |78.7 |61.8 |62.2 |
| ssd |88.9 |85.6 |87.0 |75.8 |83.1 |79.1 |76.9 |66.8 |63.4 |62.5 |
| opencv |88.2 |84.2 |87.3 |73.0 |84.4 |83.8 |81.1 |66.4 |65.5 |59.6 |
| skip |92.0 |64.1 |90.6 |56.6 |69.0 |75.1 |81.4 |57.4 |60.8 |60.7 |

## Performance Matrix for euclidean while alignment is False

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |96.1 |92.8 |95.7 |84.1 |88.3 |83.2 |78.6 |70.8 |67.4 |64.3 |
| mtcnn |95.9 |92.5 |95.5 |81.8 |89.3 |83.2 |76.3 |70.9 |65.9 |63.2 |
| fastmtcnn |96.3 |93.0 |96.0 |82.2 |90.0 |82.7 |76.8 |71.2 |66.5 |64.3 |
| dlib |96.0 |89.0 |94.1 |82.6 |96.3 |65.6 |73.1 |75.9 |61.8 |61.9 |
| yolov8 |94.8 |90.8 |95.2 |83.2 |88.4 |77.6 |71.6 |68.9 |68.2 |66.3 |
| yunet |**97.9** |96.5 |96.3 |84.1 |91.4 |82.7 |78.2 |71.7 |65.5 |65.2 |
| centerface |97.4 |95.4 |95.8 |83.2 |90.3 |82.0 |76.5 |69.9 |65.7 |62.9 |
| mediapipe |94.9 |87.1 |93.1 |71.1 |91.9 |61.9 |73.2 |77.6 |61.7 |62.4 |
| ssd |97.2 |94.9 |96.7 |83.9 |88.6 |84.9 |82.0 |69.9 |66.7 |64.0 |
| opencv |94.1 |90.2 |95.8 |89.8 |91.2 |91.0 |86.9 |71.1 |68.4 |61.1 |
| skip |92.0 |64.1 |90.6 |56.6 |69.0 |75.1 |81.4 |57.4 |60.8 |60.7 |

## Performance Matrix for euclidean_l2 while alignment is True

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |**98.4** |96.4 |95.8 |96.6 |89.1 |90.5 |92.4 |69.4 |67.7 |64.4 |
| mtcnn |**97.6** |96.8 |95.9 |96.0 |90.0 |89.8 |90.5 |70.2 |66.4 |64.0 |
| fastmtcnn |**98.1** |97.2 |95.8 |96.4 |91.0 |89.5 |90.0 |69.4 |67.4 |64.1 |
| dlib |97.0 |92.6 |94.5 |95.1 |96.4 |63.3 |69.8 |75.8 |66.5 |59.5 |
| yolov8 |97.3 |95.7 |95.0 |95.5 |88.8 |88.9 |91.9 |68.7 |67.5 |66.0 |
| yunet |**97.9** |97.4 |96.0 |96.7 |91.6 |89.1 |91.0 |70.9 |66.5 |63.6 |
| centerface |**97.7** |96.8 |95.7 |96.5 |90.9 |87.5 |89.3 |68.9 |67.8 |64.0 |
| mediapipe |96.1 |90.6 |92.9 |90.3 |92.6 |64.4 |75.4 |78.7 |64.7 |63.0 |
| ssd |88.7 |87.5 |87.0 |86.2 |83.3 |82.2 |84.6 |66.8 |64.1 |62.6 |
| opencv |87.6 |84.8 |87.3 |84.6 |84.0 |85.0 |83.6 |66.4 |63.8 |60.9 |
| skip |91.4 |67.6 |90.6 |57.2 |69.3 |78.4 |83.4 |57.4 |62.6 |61.6 |

## Performance Matrix for euclidean_l2 while alignment is False

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |**98.0** |95.9 |95.7 |95.7 |88.4 |89.5 |90.6 |70.8 |67.7 |64.6 |
| mtcnn |**97.8** |96.2 |95.5 |95.9 |89.2 |88.0 |91.1 |70.9 |67.0 |64.0 |
| fastmtcnn |**97.7** |96.6 |96.0 |95.9 |89.6 |87.8 |89.7 |71.2 |67.8 |64.2 |
| dlib |96.5 |89.9 |94.1 |93.8 |95.6 |63.0 |75.0 |75.9 |62.6 |61.8 |
| yolov8 |**97.7** |95.8 |95.2 |95.0 |88.1 |88.7 |89.8 |68.9 |68.9 |65.3 |
| yunet |**98.3** |96.8 |96.3 |96.1 |91.7 |88.0 |90.5 |71.7 |67.6 |63.2 |
| centerface |97.4 |96.3 |95.8 |95.8 |90.2 |86.8 |89.3 |69.9 |68.4 |63.1 |
| mediapipe |96.3 |90.0 |93.1 |89.3 |91.8 |65.6 |74.6 |77.6 |64.9 |61.6 |
| ssd |**97.9** |97.0 |96.7 |96.6 |89.4 |91.5 |93.0 |69.9 |68.7 |64.9 |
| opencv |96.2 |92.9 |95.8 |93.2 |91.5 |93.3 |91.7 |71.1 |68.3 |61.6 |
| skip |91.4 |67.6 |90.6 |57.2 |69.3 |78.4 |83.4 |57.4 |62.6 |61.6 |

## Performance Matrix for cosine while alignment is True

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |**98.4** |96.4 |95.8 |96.6 |89.1 |90.5 |92.4 |69.4 |67.7 |64.4 |
| mtcnn |**97.6** |96.8 |95.9 |96.0 |90.0 |89.8 |90.5 |70.2 |66.3 |63.0 |
| fastmtcnn |**98.1** |97.2 |95.8 |96.4 |91.0 |89.5 |90.0 |69.4 |67.4 |63.6 |
| dlib |97.0 |92.6 |94.5 |95.1 |96.4 |63.3 |69.8 |75.8 |66.5 |58.7 |
| yolov8 |97.3 |95.7 |95.0 |95.5 |88.8 |88.9 |91.9 |68.7 |67.5 |65.9 |
| yunet |**97.9** |97.4 |96.0 |96.7 |91.6 |89.1 |91.0 |70.9 |66.5 |63.5 |
| centerface |**97.7** |96.8 |95.7 |96.5 |90.9 |87.5 |89.3 |68.9 |67.8 |63.6 |
| mediapipe |96.1 |90.6 |92.9 |90.3 |92.6 |64.3 |75.4 |78.7 |64.8 |63.0 |
| ssd |88.7 |87.5 |87.0 |86.2 |83.3 |82.2 |84.5 |66.8 |63.8 |62.6 |
| opencv |87.6 |84.9 |87.2 |84.6 |84.0 |85.0 |83.6 |66.2 |63.7 |60.1 |
| skip |91.4 |67.6 |90.6 |54.8 |69.3 |78.4 |83.4 |57.4 |62.6 |61.1 |

## Performance Matrix for cosine while alignment is False

| | Facenet512 |Facenet |VGG-Face |ArcFace |Dlib |GhostFaceNet |SFace |OpenFace |DeepFace |DeepID |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinaface |**98.0** |95.9 |95.7 |95.7 |88.4 |89.5 |90.6 |70.8 |67.7 |63.7 |
| mtcnn |**97.8** |96.2 |95.5 |95.9 |89.2 |88.0 |91.1 |70.9 |67.0 |64.0 |
| fastmtcnn |**97.7** |96.6 |96.0 |95.9 |89.6 |87.8 |89.7 |71.2 |67.8 |62.7 |
| dlib |96.5 |89.9 |94.1 |93.8 |95.6 |63.0 |75.0 |75.9 |62.6 |61.7 |
| yolov8 |**97.7** |95.8 |95.2 |95.0 |88.1 |88.7 |89.8 |68.9 |68.9 |65.3 |
| yunet |**98.3** |96.8 |96.3 |96.1 |91.7 |88.0 |90.5 |71.7 |67.6 |63.2 |
| centerface |97.4 |96.3 |95.8 |95.8 |90.2 |86.8 |89.3 |69.9 |68.4 |62.6 |
| mediapipe |96.3 |90.0 |93.1 |89.3 |91.8 |64.8 |74.6 |77.6 |64.9 |61.6 |
| ssd |**97.9** |97.0 |96.7 |96.6 |89.4 |91.5 |93.0 |69.9 |68.7 |63.8 |
| opencv |96.2 |92.9 |95.8 |93.2 |91.5 |93.3 |91.7 |71.1 |68.1 |61.1 |
| skip |91.4 |67.6 |90.6 |54.8 |69.3 |78.4 |83.4 |57.4 |62.6 |61.1 |

# Citation

Please cite deepface in your publications if it helps your research - see [`CITATIONS`](https://github.com/serengil/deepface/blob/master/CITATION.md) for more details. Here is its BibTex entry:

```BibTeX
@article{serengil2024lightface,
  title         = {A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules},
  author        = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  journal       = {Bilisim Teknolojileri Dergisi},
  volume        = {17},
  number        = {2},
  pages         = {95-107},
  year          = {2024},
  doi           = {10.17671/gazibtd.1399077},
  url           = {https://dergipark.org.tr/en/pub/gazibtd/issue/84331/1399077},
  publisher     = {Gazi University}
}
```
