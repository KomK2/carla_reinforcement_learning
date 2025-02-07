# Reinforcement Learning for Autonomous Vehicles: Implementing PPO in CARLA

## 🚗 Overview  
This project explores the application of **Proximal Policy Optimization (PPO)** for autonomous vehicle control within the **CARLA simulator**. By leveraging **reinforcement learning (RL)**, we aim to develop an intelligent agent capable of navigating complex urban environments through trial-and-error learning.

## 📌 Features  
- **CARLA Simulation:** Uses the high-fidelity **CARLA** simulator for realistic driving scenarios.  
- **Proximal Policy Optimization (PPO):** A state-of-the-art RL algorithm for continuous control.  
- **Computer Vision Integration:** Utilizes a **CNN-based** perception model to process semantic segmentation images.  
- **Robust Reward Engineering:** Encourages safe, efficient driving behavior by penalizing collisions, lane invasions, and erratic steering.  
- **Performance Metrics:** Evaluates the model based on **collision rate, lane adherence, speed regulation, and generalization ability**.  
- **TensorBoard Visualization:** Tracks model training and performance trends.  

## 📖 Methodology  
### 1. Reinforcement Learning Framework  
- Utilizes **OpenAI Gym** interface for CARLA integration.  
- Implements **policy-based** RL using PPO.  

### 2. Observation Space  
- Semantic segmentation images (processed using CNN).  
- Normalized and cropped inputs for efficient learning.  

### 3. Reward Function  
- 🚫 **Collision Penalty**: -300  
- 🚫 **Lane Invasion Penalty**: -300  
- 🚗 **Speed Maintenance Reward**: Encourages maintaining an optimal speed of 30 km/h.  

### 4. Training Strategy  
- Trained for **2 million timesteps** with **TensorBoard monitoring**.  
- **Transfer learning** used for efficient convergence.  

### 5. Evaluation Metrics  
- Mean reward per episode.  
- Collision and lane invasion count.  
- Mean distance traveled and speed stability.  

## 🛠️ Installation  
### Prerequisites  
- **Python 3.8+**  
- **CARLA Simulator**  
- **PyTorch**  
- **Gym**  
- **TensorFlow (for TensorBoard monitoring)**  

### Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/reinforcement-learning-carla.git
   cd reinforcement-learning-carla
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Start CARLA simulator:  
   ```bash
   ./CarlaUE4.sh
   ```
4. Train the RL agent:  
   ```bash
   python train.py
   ```
5. Evaluate the trained model:  
   ```bash
   python inference.py
   ```

## 📊 Results  
- **Mean Speed Stability:** The trained agent effectively balances speed and safety.  
- **Collision-Free Navigation:** Achieved through well-engineered reward functions.  
- **Generalization Capability:** Performance was tested in multiple CARLA maps.  

## 🔬 Experiments  
- **Baseline Comparisons:** Evaluated performance with and without CNN-based preprocessing.  
- **Hyperparameter Tuning:** Adjusted timesteps and action space granularity for optimal learning.  
- **Failure Cases:** Identified challenges in intersection handling and unseen environments.  

## 📺 Inference Video  
To see the trained model in action, check out the inference video below:  
[Inference Video on Google Drive](https://drive.google.com/file/d/11mpc7Nw5Pek24NwhTL0JRmaeZgCkf_1F/view?usp=sharing)


<iframe src="https://www.youtube.com/watch?v=oqxAJKy0ii4" width="800" height="450"></iframe>


## 🔮 Future Work  
- **Transfer learning for real-world deployment.**  
- **Incorporate additional sensor modalities (LiDAR, RADAR).**  
- **Improve policy generalization to unseen environments.**  
- **Optimize for real-time execution on edge devices.**  

## 📜 References  
- CARLA Simulator: [CARLA GitHub](https://github.com/carla-simulator/carla)  
- PPO Algorithm: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)  

---
