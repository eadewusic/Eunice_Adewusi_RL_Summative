# 🚦 Rwanda Traffic Junction - Reinforcement Learning Optimization System

This project develops and evaluates reinforcement learning agents for intelligent traffic light control at busy junctions in Rwanda. Through comprehensive experimentation with 4 different RL algorithms and 17 distinct configurations, it is demonstrated that AI agents can significantly outperform traditional traffic management methods.

![image](./visualization/images/3d-traffic-demo.png)

**Mission: Replace road wardens with intelligent agents** - A comprehensive reinforcement learning study to optimize traffic flow at busy junctions in Rwanda using state-of-the-art RL algorithms.

![RL Algorithms](https://img.shields.io/badge/Algorithm_Families-4-orange)
![Configurations](https://img.shields.io/badge/Configurations-17-red)
![Training Progress](https://img.shields.io/badge/Training_Steps-300,000+-green)
![Success Rate](https://img.shields.io/badge/Success_Rate-56.3%25-blue)
![Best Performance](https://img.shields.io/badge/Best_Improvement-+74.7%25-gold)

### Real-World Impact
- **Location**: Kigali, Rwanda traffic junctions
- **Problem**: Inefficient manual traffic management by road wardens
- **Solution**: Intelligent RL agents for automated traffic optimization
- **Achievement**: **+74.7% improvement** over random baseline with PPO

## Key Achievements

### **Best Performance: PPO_AGGRESSIVE**
- **Reward**: -151.85 ± 22.42
- **Improvement**: +74.7% over random baseline
- **Classification**: **ELITE TIER** performance
- **Training Efficiency**: Excellent (200,000 timesteps, 5-7 minutes)

### **Comprehensive Study Results**
- **17 Total Configurations Tested** (16 trained + 1 random baseline)
- **4 Algorithm Families**: PPO, REINFORCE, Actor-Critic, DQN
- **Success Rate**: 56.3% (9/16 trained configurations beat random baseline)
- **Elite Tier**: 4 configurations (all PPO variants)
- **Performance Range**: -151.85 to -14970.75 (98.7x difference!)
- **Training Methods**: Mixed (timesteps for some, episodes for others)
- **Episodes Logged:** 300,000+ training steps across mixed methodologies

### **Algorithm Performance Ranking**
1. **PPO Family**: 100% success rate (4/4 configs successful)
2. **REINFORCE Family**: 75% success rate (3/4 configs successful)  
3. **Actor-Critic Family**: 50% success rate (2/4 configs successful)
4. **DQN Family**: 0% success rate (0/3 configs above baseline)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/eadewusic/Eunice_Adewusi_RL_Summative.git
cd Eunice_Adewusi_RL_Summative

# Create virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Setup Check
python main.py --info

# Create Random Action Demo

python visualization/demo_random.py

# Run specific training

python main.py --train-dqn
python main.py --train-reinforce
python main.py --train-ppo
python main.py --train-ac

# Run specific Hyperparameter Tuning training (best algorithm)
python main.py --train ppo --config aggressive

# Run complete evaluation (generates all 16 output files)
python main.py --evaluate

# Generate video demonstration
python traffic_video_demo.py

# Run 2D visualization
python visualization/2d_traffic_simulation.py

# Run 3D visualization
python visualization/3d_traffic_simulation.py

# Run this for newly installed packages
pip freeze | ForEach-Object { $_.Split('==')[0] } > requirements.txt
```

## Results Summary

### **Top Performing Configurations**
| Rank | Configuration | Final Reward | Std Dev | Improvement | Tier |
|------|---------------|--------------|---------|-------------|------|
| 1st | PPO_AGGRESSIVE | -151.85 | ±22.42 | +74.7% | **ELITE** |
| 2nd | PPO_CONSERVATIVE | -153.60 | ±18.31 | +74.4% | **ELITE** |
| 3rd | PPO_HIGH_ENTROPY | -154.00 | ±22.66 | +74.3% | **ELITE** |
| 4th | PPO_MAIN_TRAINING | -158.25 | ±20.41 | +73.6% | **ELITE** |
| 5th | ACTOR_CRITIC_BALANCED | -182.70 | ±17.83 | +69.5% | **EXCELLENCE** |

### **Performance vs. Random Baseline**
- **Random Baseline**: -599.80 ± 223.80
- **Best Achievement**: -151.85 ± 22.42 (PPO Aggressive, +74.7% improvement)
- **Worst Failure**: -14970.75 ± 8.10 (Actor-Critic Aggressive, -2396% degradation)
- **Configurations Beating Random**: 9 out of 16 trained configurations (56.3%)

### **Complete Performance Analysis - All 17 Configurations**

| Rank | Algorithm Family | Configuration | Final Reward | Std Dev | Improvement vs Random | Status |
|------|------------------|---------------|--------------|---------|----------------------|---------|
| **1** | **PPO** | **AGGRESSIVE** | **-151.85** | **±22.42** | **+74.7%** | **CHAMPION** |
| **2** | **PPO** | **CONSERVATIVE** | **-153.60** | **±18.31** | **+74.4%** | **EXCELLENT** |
| **3** | **PPO** | **HIGH_ENTROPY** | **-154.00** | **±22.66** | **+74.3%** | **EXCELLENT** |
| **4** | **PPO** | **MAIN_TRAINING** | **-158.25** | **±20.41** | **+73.6%** | **EXCELLENT** |
| **5** | **Actor-Critic** | **BALANCED** | **-182.70** | **±17.83** | **+69.5%** | **VERY GOOD** |
| **6** | **Actor-Critic** | **CONSERVATIVE** | **-188.30** | **±22.73** | **+68.6%** | **VERY GOOD** |
| **7** | **REINFORCE** | **MAIN_TRAINING** | **-188.95** | **±25.29** | **+68.5%** | **VERY GOOD** |
| **8** | **REINFORCE** | **CONSERVATIVE** | **-191.10** | **±23.97** | **+68.1%** | **VERY GOOD** |
| **9** | **REINFORCE** | **MODERATE** | **-242.80** | **±27.87** | **+59.5%** | **GOOD** |
| **10** | **Random** | **BASELINE** | **-599.80** | **±223.80** | **0% (Baseline)** | **REFERENCE** |
| **11** | **DQN** | **AGGRESSIVE** | **-778.60** | **±458.27** | **-29.8%** | **BELOW RANDOM** |
| **12** | **DQN** | **MAIN_TRAINING** | **-1371.20** | **±644.80** | **-128.6%** | **POOR** |
| **13** | **DQN** | **CONSERVATIVE** | **-7312.85** | **±197.58** | **-1119%** | **CRITICAL FAILURE** |
| **14** | **REINFORCE** | **STANDARD** | **-7967.75** | **±12.50** | **-1228%** | **CRITICAL FAILURE** |
| **15** | **Actor-Critic** | **BASELINE** | **-9238.35** | **±113.48** | **-1440%** | **CATASTROPHIC** |
| **16** | **Actor-Critic** | **AGGRESSIVE** | **-14970.75** | **±8.10** | **-2396%** | **CATASTROPHIC** |

### The Shocking Discovery: Random Actions Beat 6/16 Trained Models

One of the most significant findings of this project: **Random actions (-599.80) outperformed 6 out of 16 sophisticated RL configurations**, including:

- **All 3 DQN configurations** (-778.60 to -7312.85)
- **REINFORCE Standard** (-7967.75) 
- **Actor-Critic Baseline** (-9238.35)
- **Actor-Critic Aggressive** (-14970.75)

This reveals both the genuine difficulty of traffic optimization and the critical importance of proper hyperparameter tuning in RL deployment.

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU training)
- OpenCV (for video generation)
- PyBullet (for 3D simulations)

## RL Algorithms

### 1. **PPO (Proximal Policy Optimization)**
- **Status**: **CHAMPION** (All 4 configs successful)
- **Best Config**: Aggressive (+74.7% improvement)
- **Strengths**: Stable, efficient, excellent hyperparameter robustness
- **Training Time**: 13-49 episodes
- **Use Case**: **Recommended for production deployment**

### 2. **REINFORCE (Policy Gradient)**
- **Status**: **SOLID PERFORMER** (3/4 configs successful)
- **Best Config**: Main Training (+68.5% improvement)
- **Strengths**: Good convergence, handles complex scenarios
- **Training Time**: 750-1000 episodes
- **Use Case**: Secondary option for research environments

### 3. **Actor-Critic (Hybrid)**
- **Status**: **MIXED RESULTS** (2/4 configs successful)
- **Best Config**: Balanced (+69.5% improvement)
- **Strengths**: Can achieve good results with proper tuning
- **Weakness**: Extremely sensitive to hyperparameters
- **Use Case**: Experimental/research purposes only

### 4. **DQN (Deep Q-Network)**
- **Status**: **UNDERPERFORMER** (0/3 configs successful)
- **Best Config**: Aggressive (-29.8% degradation)
- **Issues**: Poor adaptation to traffic environment
- **Training Time**: 100,000 timesteps (5-6 minutes, inefficient)
- **Use Case**: Not recommended for this domain

## Training & Evaluation

### **Complete Experimental Scope**
- **Total Configurations**: 17 (16 trained algorithms + 1 random baseline)
- **Algorithm Families**: 4 (PPO, REINFORCE, Actor-Critic, DQN)
- **Hyperparameter Variants**: Extensive grid search across key parameters
- **Training Methodology**: Mixed approach (timesteps for PPO/DQN, episodes for REINFORCE/Actor-Critic)
- **Performance Spread**: 98.7x difference between best and worst performers

### **Evaluation Metrics**
- **Primary**: Episode reward (traffic flow efficiency)
- **Secondary**: Vehicle processing rate, queue lengths, waiting times
- **Stability**: Reward variance and convergence analysis  
- **Efficiency**: Training time vs. performance trade-offs

### **Training Specifications by Algorithm**
- **PPO**: 200,000 timesteps (5-7 minutes training)
- **REINFORCE**: 850-1000 episodes (51 seconds - 2.3 minutes)
- **Actor-Critic**: 1000 episodes (1.5-23 minutes depending on config)
- **DQN**: 100,000 timesteps (5-6 minutes)

### **Hyperparameter Configurations**

#### PPO Variants
- **Aggressive**: High learning rate, large policy updates
- **Conservative**: Stable learning, small incremental improvements
- **High Entropy**: Exploration-focused for traffic pattern discovery
- **Main Training**: Balanced baseline configuration

#### REINFORCE Variants  
- **Main Training**: Standard policy gradient approach
- **Conservative**: Reduced learning rate for stability
- **Moderate**: Balanced exploration-exploitation
- **Standard**: Classical REINFORCE implementation

#### Actor-Critic Variants
- **Balanced**: Equal actor-critic learning rates
- **Conservative**: Critic-focused learning
- **Baseline**: Standard implementation
- **Aggressive**: Actor-focused rapid learning

#### DQN Variants
- **Main Training**: Standard DQN with experience replay
- **Aggressive**: High learning rate, frequent updates
- **Conservative**: Stable Q-learning approach

## Video Demonstrations

![image](/visualization/images/2d-traffic-demo.png)

### **2D Traffic Demo**
- **File**: `rwanda_traffic_2d_demo.mp4`
- **Duration**: 6.25 seconds (125 frames)
- **Performance**: VERY_GOOD (-128.33 average reward)
- **Model**: PPO_AGGRESSIVE (best performer)
- **Episodes**: 3 complete scenarios

### **3D Traffic Demo**  
- **File**: `traffic_3d_demo_20250801_200808.mp4`
- **Duration**: 3.05 seconds (61 frames)
- **Performance**: GOOD (-152.00 average reward)
- **Features**: Realistic 3D environment with PyBullet physics
- **HUD**: Real-time metrics display

### **Demo Insights**
- **Smart Decision Making**: Agent adapts to traffic patterns
- **Emergency Handling**: Priority vehicle processing
- **Efficiency**: Reduced queue times and improved flow
- **Realistic Scenarios**: Rwanda traffic patterns simulated

## Visualizations

### **Training Analysis (16 Charts Generated)**

#### **Learning Curves**
- `best_training_curves.png` - Top 4 configurations
- `all_configuration_curves.png` - Complete 17-config analysis
- `cumulative_reward_curves.png` - Episode reward progression

#### **Stability Analysis**
- `training_stability_analysis.png` - 6-panel stability metrics
- `objective_function_curves.png` - DQN Q-values & policy analysis

#### **Performance Comparisons**
- `performance_comparison_chart.png` - Algorithm family comparison
- `statistical_analysis_chart.png` - Confidence intervals & statistics
- `comprehensive_17_config_comparison.png` - Complete study overview

#### **Advanced Analytics**
- `scenario_analysis_chart.png` - Traffic scenario performance
- `performance_matrix_heatmap.png` - Configuration success matrix

## Data & Logging

### **Training Logs**
- **Episode Metrics**: Reward, length, convergence tracking
- **Step-by-Step Data**: Action selection, state transitions
- **Performance Summaries**: Training time, final results
- **Hyperparameter Records**: Complete configuration tracking

### **Evaluation Data**
- **CSV Exports**: Machine-readable performance data
- **JSON Results**: Complete structured evaluation results
- **Statistical Analysis**: Confidence intervals, significance tests
- **Comparative Studies**: Cross-algorithm performance analysis

### **Data Quality**
- **300,000+ Training Steps**: Comprehensive training coverage across mixed methodologies
- **Multi-scenario Testing**: Various traffic conditions
- **Reproducible Results**: Seed-based consistency
- **Validated Metrics**: Statistical significance testing

## Rwanda Context

### **Real-World Application**
- **Location**: Kigali traffic junctions
- **Problem**: Manual road warden inefficiency
- **Traffic Patterns**: Rush hours (7-9 AM, 5-7 PM)
- **Vehicle Types**: Cars, buses (public transport), motorcycles
- **Emergency Priority**: Ambulances and emergency vehicles

## Technical Details

### **Environment Specifications**
- **State Space**: 113-dimensional observation vector
- **Action Space**: 9 discrete traffic control actions
- **Reward Function**: Multi-objective (flow + safety + efficiency)
- **Episode Length**: Variable (terminated by traffic conditions)

### **Action Space**
**Type:** Discrete - gym.spaces.Discrete(9)

**Traffic Light Control Actions:**
- **Action 0: Extend North-South Green** - Add 10 seconds to the current NS green phase (max 90s total)
- **Action 1: Extend East-West Green** - Add 10 seconds to the current EW green phase (max 90s total)
- **Action 2: Switch to North-South Green** - Immediately change to NS green with 30s default timing
- **Action 3: Switch to East-West Green** - Immediately change to EW green with 30s default timing
  
**Emergency and Special Control Actions:**
- **Action 4: Emergency Priority Override** - Activate emergency vehicle protocol with immediate direction switching
- **Action 5: All Red Transition** - Set all lights to red for 5-second safe transition period
- **Action 6: Reset Timer** - Reset current phase to 30-second standard timing
  
**Adaptive Timing Actions:**
- **Action 7: Short Green Cycle** - Set 15-second cycle for light traffic conditions
- **Action 8: Extended Green Cycle** - Set 60-second cycle for heavy traffic conditions

### **Model Architectures**
- **PPO**: Separate actor-critic networks (256-128-64 each) - Stable Baselines3
- **DQN**: Deep Q-Network (256-128-64 architecture) - Stable Baselines3  
- **REINFORCE**: Policy network (128-128-64) + baseline value network for variance reduction
- **Actor-Critic**: Custom dual networks - 128-128-64 actor (policy) and 128-128-64 critic (value function)

### **Training Configuration**

**PPO Configurations (4 variants tested)**:
- Learning rates: 0.0001 to 0.001
- Batch sizes: 32 to 128  
- Epochs: 5 to 20
- All used Stable Baselines3 with 200,000 timesteps

**DQN Configurations (3 variants tested)**:
- Learning rates: 0.0001 to 0.001
- Batch sizes: 32 to 128
- Buffer size: 50,000 (consistent across all)
- Exploration (ε): 1.0 → 0.02/0.05/0.1 depending on config

**REINFORCE Configurations (4 variants tested)**:
- Learning rates: 0.0003 to 0.0008
- All used γ=0.99, baseline normalization
- Episodes: 850-1000 (with early stopping for best config)

**Actor-Critic Configurations (4 variants tested)**:
- Actor learning rates: 0.0005 to 0.002
- Critic learning rates: 0.001 to 0.003  
- Update frequency: 10 steps (consistent)
- Gradient clipping: 0.5 max norm

---

*This project demonstrates that intelligent RL agents can achieve up to **74.7% improvement** over traditional traffic management, but successful deployment critically depends on proper algorithm and configuration selection.*
