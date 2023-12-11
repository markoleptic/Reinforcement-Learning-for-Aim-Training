# Reinforcement Learning for Aim Training

This repository implements various machine learning algorithms from scratch using Python. 

The results of this project help create the reinforcement learning available in [BeatShot](https://github.com/markoleptic/BeatShot) today. You can see how this was implmented in C++ [here](https://github.com/markoleptic/BeatShot/blob/develop/Source/BeatShot/Private/Target/ReinforcementLearningComponent.cpp).

## Files
- **Algos.py:** Implements Sarsa and Q-Learning reinforcement learning algorithms.
- **ExperimentLoop.py:** Creates the custom Gymnasium environment, ML_RL_Env and uses algorithms from Algos.py.
- **FileModHandler.py:** Watches for changes to a given file and executes a callback when modified.
- **LiveGameExperiment.py:** Uses FileModHandler to watches for changes to the `Accuracy.csv` file, and updates the QTable when modified. This was created so that [BeatShot](https://github.com/markoleptic/BeatShot) could write to this file.
- **ML_RL_Env.py:** Creates a custom Gymnasium environment using a static player accuracy matrix. PyGame is built into the environment, and you can watch the environment change rewards at locations during iteration by specifying `render_mode='human'`.
- **Report.pdf:** A report for the project containing a full explanation, analysis, and conclusions for the project.

<table>
  <h2 align="center"><b>Example Results</b></h2>
    <tr>
    <td width=50%>
        <img src="https://github.com/markoleptic/ReinforcementLearning-for-AimTraining/assets/86213229/81a59900-16b6-445a-aee6-2538c589b584" alt="roc">
    </td>
    <td width=50%>
        <img src="https://github.com/markoleptic/ReinforcementLearning-for-AimTraining/assets/86213229/151636de-100e-4524-97f9-7477ef21a762" alt="weights">
    </td>
  </tr>
</table>
<table>
  <h2 align="center"><b>Q-Table Heatmaps</b></h2>
  <tr>
    <td width=50%>
        <img src="https://github.com/markoleptic/ReinforcementLearning-for-AimTraining/assets/86213229/7fd203d4-afe2-4612-b7c9-13b344596de9" alt="weights">
    </td>
    <td width=50%>
        <img src="https://github.com/markoleptic/ReinforcementLearning-for-AimTraining/assets/86213229/9205dc11-c32d-47c8-9a55-be602b88a04d" alt="roc">
    </td>
  </tr>
</table>
