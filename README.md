# Retina Image Segmentation
🧠 **Retina Blood Vessel Segmentation using PCA & K-Means**

- This project implements an automated retinal blood vessel segmentation system using classical machine learning techniques. It combines image processing, unsupervised ML, a Flask API, and a Flutter mobile app to deliver an end-to-end solution for medical image analysis.

📌 **Problem Statement**

- Retinal blood vessel segmentation is a critical step in diagnosing eye diseases such as diabetic retinopathy, glaucoma, and hypertension. Manual segmentation is time-consuming and error-prone.
This project automates the process using PCA for feature reduction and K-Means clustering for vessel extraction, making it lightweight and deployable on low-resource systems.

🚀 **Key Features**

- Automated blood vessel segmentation from retinal images
- Unsupervised ML pipeline using PCA + K-Means
- Flask REST API for real-time image segmentation
- Flutter mobile app (APK) for easy image upload and result visualization
- Model evaluation using Intersection over Union (IoU)

🛠️ **Technology Stack**
- Frontend (Mobile App):	Flutter
- Backend (API):	Flask
- Machine Learning:	PCA, K-Means Clustering
- Image Processing:	OpenCV, NumPy
- Deployment:	Render (Flask API), GitHub
- Mobile Packaging:	Android APK

🔮 **Future Enhancements**

- Replace classical ML with U-Net / Deep Learning
- Cloud storage for segmentation history
- Support segmentation without mask input
- Improve performance using larger datasets

📤 **Output**

- **System Architecture**

![System Architecture](https://github.com/AtharvaDaga/Retina-Segmentation/blob/6da978a01e708c22854fea9376d397e84dfcbab8/system/system%20design.jpg)

- **Image Upload Snapshot**

![Uploading Retina Image](https://github.com/AtharvaDaga/Retina-Segmentation/blob/6da978a01e708c22854fea9376d397e84dfcbab8/system/uploading%20of%20the%20images.jpg)

- **Segmented Output**

![Segmented Retina Image](https://github.com/AtharvaDaga/Retina-Segmentation/blob/6da978a01e708c22854fea9376d397e84dfcbab8/static/segmented_output/segmented_image.jpg)
