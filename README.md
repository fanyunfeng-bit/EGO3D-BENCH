<div align="center">
  <h1>Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-view Scenes (Code Comes Soon!)</h1>
  <p><i>Benchmarking and Improving 3D Spatial Reasoning in Vision-Language Models</i></p>

<a href="https://arxiv.org/abs/2509.06266" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-red?logo=arxiv" height="20" />
</a>
<a href="https://vbdi.github.io/Ego3D-Bench-webpage/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/vbdai/Ego3D-Bench" target="_blank">
    <img alt="HF Dataset: Ego3D-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Ego3D_Bench-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
</div>

---
![Sample](figs/Fig2_v4.png)

### 📌 Key Highlights

- 📊 **Ego3D-Bench**: A benchmark of **8,600+ human-verified QA pairs** for evaluating VLMs in **ego-centric, multi-view outdoor environments**.  
- 🧠 **Ego3D-VLM**: A **post-training framework** that builds cognitive maps from global 3D coordinates, achieving **+12% QA accuracy** and **+56% distance estimation** improvements.  
- 🚀 **Impact**: Together, Ego3D-Bench and Ego3D-VLM move VLMs closer to **human-level 3D spatial understanding** in real-world settings.  

---


### ⚖️ **Ego3D-Bench**
Benchmark Overview: We introduce Ego3D-Bench, a benchmark designed to evaluate the spatial understanding of VLMs in ego-centric multi-view scenarios. Images are collected from three different datasets: NuScenes, Argoverse, and Waymo. Questions are designed to require cross-view reseasoning. We define question from the ego-perspective and from the perspective of objects in the scene. To clearly indicate the perspective of each question, we categorize them into ego-centric or object-centric. In total we have 10 questions: 8 multi-choice QAs and 2 exact number QAs. Figure 

![Sample](figs/Fig5_v2.png)

---
### 🧠 **Ego3D-VLM**
Ego3D-VLM is a post-training framework that enhances 3D spatial reasoning of VLMs. Ego3D-VLM generates cognitive map based on estimated global 3D coordinates of object in the input prompt and create a textual cognitive map baed on the input images and the question. This approach results in 12% average improvement on multi-choice QA and 56% average improvement on absolute distance estimation.
![Sample](figs/Fig1_v2.png)

---
### 📊 **Results**
![Sample](figs/Res1.png)

---
