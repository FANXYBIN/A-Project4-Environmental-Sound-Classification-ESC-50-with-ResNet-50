# A-Project 5: Environmental Sound Classification (ESC-50) with ResNet-50


This project explored **environmental sound classification** using the **ESC-50 dataset**, converting audio clips into **Mel spectrograms** and training a fine-tuned **ResNet-50 CNN model**.  
The workflow included data preprocessing, spectrogram generation, augmentation (SpecAugment), model training, and evaluation with accuracy and confusion matrix metrics.

* **Dataset:** ESC-50 (2,000 labeled audio clips, 50 sound classes)  
* **Tools:** Python, PyTorch, torchaudio, librosa, matplotlib  
* **Techniques:** Mel spectrograms, SpecAugment, custom dataset loader, transfer learning (ResNet-50)  
* **Goal:** Build an effective audio classifier by transforming audio into images for CNN-based learning.

---

### 🎼 Dataset & Preprocessing

- ESC-50 consists of **2,000 WAV files** across **50 categories** (animals, natural sounds, human noises, etc.).  
- Converted each WAV file into a **Mel spectrogram** using `librosa`, then saved as `.png` images for image-based CNN training.  
- Implemented a **custom PyTorch dataset class (SpecDataset)** to load spectrograms and labels.  
- Applied preprocessing: resizing, normalization, and **SpecAugment** (frequency & time masking).  
- Split dataset into **train / validation / test** using filename rules.

<div align="center">
  <img src="images/esc50_spectrogram_example.png" width="600"/>
  <p><em>ESC-5 Dataset.</em></p>
</div>
<div align="center">
  <img src="images/Mel Spectrogram.png" width="600"/>
  <p><em>Mel Spectrogram with frequency & time masking (SpecAugment).</em></p>
</div>
---

### 🏗️ Model Architecture & Training

- Used **ResNet-50 pretrained on ImageNet**, freezing all layers except:  
  - Layer 3  
  - Layer 4  
  - Final classifier  
- Replaced the last fully connected layer with a **50-class classifier**.  
- Set learning rate = **0.01**, with **LR decay (×0.5 every 6 epochs)**.  
- Training configured for **66 epochs** with **early stopping (patience = 6)**.  
- Saved checkpoint whenever **validation loss improved**.

<div align="center">
  <img src="images/esc50_training_curve.png" width="600"/>
  <p><em>Training vs. validation loss and accuracy trends.</em></p>
</div>

**Observations:**
- Training accuracy → **~99%**, Validation accuracy → **~80%**  
- Validation loss stabilized after ~18 epochs  
- Training–validation accuracy gap indicates **mild overfitting**, improved by SpecAugment

---

### 📊 Evaluation

- Training stopped at **epoch 41** due to early stopping  
- **Best validation accuracy:** **80.34%**  
- **Validation loss:** **0.5999**  
- **Test accuracy:** **70.94%**  
- **Test loss:** **1.0581**



Performance was strong on distinct classes (e.g., dog bark, rain) but more challenging for acoustically similar categories (e.g., engine sounds vs. machinery).

---

### ❓ Q&A

**1. Why this dataset?**  
ESC-50 is widely used for environmental sound research. It is balanced, cleanly labeled, and manageable for academic deep-learning workflows.

**2. What modifications were needed?**  
- Converted WAV files → Mel spectrogram images  
- Created a **PyTorch Dataset** to load spectrograms  
- Designed filename-based train/val/test splitting  
- Implemented preprocessing & SpecAugment  
- Built custom classifier for a 50-class CNN task

**3. What challenges did you encounter?**  
- Inconsistent filename patterns → solved using **regex**  
- Overfitting → mitigated using **SpecAugment + dropout**  
- Normalization distorted spectrogram visuals → fixed by debugging preview pipeline  
- Balancing training time vs. performance required LR tuning

**4. Would the model be deployable? Why or why not?**  
Not yet. While achieving strong accuracy (80% val, 71% test), deployment-readiness requires:  
- Real-time spectrogram generation  
- Input-output interface (microservice / app)  
- Model size optimization  
- Handling unseen audio conditions  

---

### 🧠 Skills Demonstrated

- Audio → image transformation using Mel spectrograms  
- CNN fine-tuning (ResNet-50) for audio tasks  
- PyTorch training pipelines, LR scheduling, early stopping  
- Data augmentation via SpecAugment  
- Model evaluation with confusion matrix  


