# AudioSpace: Generating Spatial Audio from 360-Degree Video  

✨🔊 Transform your 360-degree videos into immersive spatial audio! 🌍🎶  

<img src="assets/figure1-a.png" width="45%"> <img src="assets/figure1-b.png" width="45%">  

PyTorch Implementation of **AudioSpace**, a model for generating spatial audio from 360-degree videos.  

Due to anonymization requirements, ##**our model will be available on Hugging Face soon.**  

---

## 🎬 Quick Start  
We provide an example of how you can perform inference using AudioSpace.  

### 🏃 Inference with Pretrained Model  
To run inference, follow these steps:  

1️⃣ **Navigate to the root directory.** 📂  
2️⃣ **Create the Inference Environment.**  

To set up the environment, ensure you have **Python >= 3.8.20** installed. Then, run the following commands:  

```bash
pip install -r requirements.txt
pip install git+https://github.com/patrick-kidger/torchcubicspline.git
```
 
3️⃣ **Run inference with the provided script:**  
   ```bash
   bash demo.sh video_path checkpoint_dir cuda_id
   ```  
💡 *You can also modify `demo.sh` to change the output directory.* The `cases` folder contains some sample 360-degree videos in the equirectangular format—make sure your videos follow the same format! 🎥✨  

---

## 📊 Evaluation  
To evaluate generated audio, follow these steps:  

1️⃣ **Navigate to** `metrics/evaluate`. 📂  
2️⃣ **Create the evaluation environment:**  
    To set up the environment, ensure you have **Python >= 3.10.15** installed. Then, run the following commands: 
   ```bash
   pip install -r requirements.txt
   ```  
3️⃣ **Modify `eval.sh`** to specify:  
   - 🎧 The path to real (ground truth) audio files  
   - 🎵 The path to generated audio files  
   - 📜 The text file containing the names of the audio files to be compared  

Before running `eval.sh`, make sure to fill in these variables inside the script:  
```bash
reference_path=""    # Path to ground truth audios  
split_path="samples.txt"  # Path to split txt  
result_path=""    # Path to save results  
```  

4️⃣ **Run evaluation:**  
   ```bash
   bash eval.sh cuda_id
   ```  

✨ **Done!** 🎉 Now you can compare the generated spatial audio with the ground truth! 🚀🔍  

---

💡 *Have fun experimenting with AudioSpace! 🛠️💖*

## References and Licensing 

The model and inference components are adapted from [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools). The corresponding license can be found in `audiospace/LICENSE`. Stable Audio Tools also includes dependencies that require specific licenses, which are available in the `audiospace/LICENSES` directory.  

The **metrics** components are adapted from [PaSST](https://github.com/kkoutini/PaSST) and [OpenL3](https://github.com/marl/openl3). Their respective licenses can be found in `metrics/LICENSES`. For a detailed breakdown, refer to `metrics/LICENSES/README.md`.  

## Acknowledgments  
We would like to express our gratitude to the authors and contributors of **Stable Audio Tools, PaSST, and OpenL3** for their invaluable work, which has greatly influenced and supported this project. Their contributions to the open-source community have been instrumental in advancing audio-related research and applications.  

