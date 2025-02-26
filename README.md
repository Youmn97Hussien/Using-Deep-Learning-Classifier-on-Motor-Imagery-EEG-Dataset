# 🧠 Using-Deep-Learning-Classifier-on-Motor-Imagery-EEG-Dataset

## 📊 Data Collection
EEG data was collected during sleep, capturing brain signals for motor imagery tasks.

## 💤 Pre-Sleep and Post-Sleep Data
The dataset includes both pre-sleep and post-sleep data:
- **Pre-sleep data:** Used for training and validation.
- **Post-sleep data:** Reserved for testing.

## 🧩 Multi-Channel Data
The EEG dataset consists of multi-channel brain signals, providing a comprehensive view of neural activity during motor imagery tasks.

### 📂 Dataset Source
The dataset used for this project was obtained from the **Open Science Framework (OSF)**, specifically from the dataset titled **"REM sleep TMR in human REM sleep elicits detectable reactivation with TMR"** and **"TMR in human REM sleep elicits detectable reactivation part2"**. It is available at:
- [Dataset URL 1](https://osf.io/wmyae/)
- [Dataset URL 2](https://osf.io/fq7v5/)

## 🛠️ Data Preprocessing
The data preprocessing steps are performed using **Matlab**:
- **Artifact Removal:** EOG1 and EOG2 channels were removed to eliminate their artifacts.
- **Baseline Correction:** The 200 milliseconds before the stimulus were considered as the baseline to apply baseline correction.
- **Clipping:** The recorded length was clipped to [-0.2, 0.9] seconds, focusing on the period of the trial.
- **Band-Pass Filtering:** A band-pass filter was applied, including frequencies within the range [0.1, 40] Hz.

## 🧑‍💻 Classification Model
The classification model is implemented using **Python**, leveraging deep learning techniques.

### Key Techniques:
- **Deep Learning Classifier:** Utilized for analyzing EEG data.
- **Motor Imagery Tasks:** Focused on neural activity during motor imagery.

## 📄 Reference Paper
The data and methodology of this project are based on the research paper:
- **Title:** Targeted memory reactivation in human REM sleep elicits detectable reactivation.
- **Authors:** Mahmoud EA Abdellahi, Anne CM Koopman, Matthias S Treder, Penelope A Lewis.
- **Published in:** eLife, 2023.
- **Paper URL:** [https://elifesciences.org/articles/84324].
