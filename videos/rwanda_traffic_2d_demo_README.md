# Rwanda Traffic Junction RL Agent Demo

## Overview
This video demonstration showcases a Reinforcement Learning agent trained to control traffic lights at a busy junction in Rwanda. The agent uses PPO (Proximal Policy Optimization) to learn optimal traffic management strategies.

## Demo Information
- **Created**: 2025-08-01T19:42:59.954973
- **Agent Type**: PPO Model
- **Model Path**: `models/ppo_tuning/aggressive/ppo_aggressive_final.zip`
- **Video Output**: `videos/rwanda_traffic_2d_demo.mp4`
- **Summary Data**: `rwanda_traffic_2d_demo_summary.json`

## Video Specifications
- **Resolution**: 1200x800 pixels
- **Frame Rate**: 20 FPS
- **Duration**: 6.2 seconds
- **Total Frames**: 125
- **File Size**: 2.2 MB

## Episode Performance

| Episode | Steps | Total Reward | Performance |
|---------|-------|--------------|-------------|
| 1 | 25 | -149.00 | GOOD |
| 2 | 23 | -110.00 | GOOD |
| 3 | 17 | -126.00 | GOOD |

## Summary Statistics
- **Average Reward**: -128.33
- **Best Episode**: Episode 2 (-110.00 reward)
- **Worst Episode**: Episode 1 (-149.00 reward)
- **Performance Rating**: VERY_GOOD

## Actions Available
The agent can choose from 9 different actions:
1. **Extend NS Green** - Keep North-South traffic flowing longer
2. **Extend EW Green** - Keep East-West traffic flowing longer  
3. **Switch to NS** - Change to North-South green phase
4. **Switch to EW** - Change to East-West green phase
5. **Emergency Priority** - Activate emergency vehicle protocol
6. **All Red** - Stop all traffic temporarily
7. **Reset Timer** - Reset current phase timer
8. **Short Cycle** - Use shorter light cycle
9. **Extended Cycle** - Use longer light cycle

## Environment Details
- **State Space**: Vehicle positions, waiting times, light phases
- **Reward Function**: Based on traffic flow efficiency and waiting times
- **Traffic Pattern**: Realistic Rwanda urban traffic simulation

## How to Run
```bash
python traffic_video_demo.py
```

## Files Generated
- `rwanda_traffic_2d_demo.mp4` - Main demonstration video
- `rwanda_traffic_2d_demo_summary.json` - Detailed statistics and metadata
- `README.md` - This documentation file

## Notes
SUCCESS: Model successfully loaded and used for intelligent traffic control

---
*Generated automatically by Rwanda Traffic RL Demo System*
