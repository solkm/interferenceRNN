
# Overview

This repo contains code for the recurrent neural network (RNN) modeling 
component of the preprint:
    
>Task interference as a neuronal basis for the cost of cognitive flexibility. Cheng Xue, Sol K. Markman, Ruoyi Chen, Lily E. Kramer, Marlene R. Cohen. bioRxiv 2024.03.04.583375; doi: https://doi.org/10.1101/2024.03.04.583375

Published in final form as:

>(Currently in review)
>See the `manuscript` folder for the most recent version compatible with the figure descriptions below.

It includes code to train new models and simulate new data, the behavioral data used to train the models, the weights for the models used in the paper, and analysis and plotting code to generate the figures in the paper (prior to editing in Adobe Illustrator).

To download the exact simulated data used for the figures in the paper, see the following Box folder for large files:  
**link here**

# Resources

This project was developed using the PsychRNN package:
>Documentation: https://psychrnn.readthedocs.io/en/latest/
>GitHub: https://github.com/murraylab/PsychRNN
>Paper: https://www.eneuro.org/content/8/1/ENEURO.0427-20.2020. 

You do not need to install PsychRNN separately--all code needed for this project is included here in the folder `src/psychrnn`. However, the PsychRNN documentation may be helpful for understanding the backend code. The code included here has been modified to support TensorFlow 2.x.

For the distance covariance analysis (DCA), code was modified from: https://github.com/BenjoCowley/dca (paper: https://proceedings.mlr.press/v54/cowley17a/cowley17a.pdf). It is included here in the file `src/dca.py`.

# Dependencies

The `environment.yml` file contains the conda environment with all necessary dependencies to run the code.  
The code has been most recently tested with TensorFlow 2.17.0.

# Generating figures: analysis and plotting code

The files references below are located in `src/figure_scripts`. Generated figures can be saved as pdfs in the `figs` folder.

## Figure 2 (and supplementary figures 4 & 5)

### 2B: Perceptual performance drop during training

See `perc_acc_training_hist.py` to see how perceptual accuracy during training was calculated from the correct choice model's training history.

See `perc_acc_training_weights.py` to see how perceptual accuracy during training was calculated from the monkey choice model's weights. Run the last cell in this script to plot the figure.

### 2C: Perceptual performance scatterplots

Run `plot_ppss.py` to compute accuracies and plot the figure.

See `generate_data_ppss.py` to see how the dataset with evenly sampled stimulus conditions was generated (or to generate a new dataset).

### 2D and S4: Feature decoder accuracies

See `delay_stim1decoders.py` for details on decoder training and testing, or to generate a new accuracy dataframe.

Run `plot_delay_stim1decoders.py` to generate figures S4 (all accuracies vs. time) 
and 2D (irrelevant feature decoder scatterplot).

### 2E and S5: DCA feature axes

See `dca_featureaxes.py` for details on how DCA is applied to compute an axis for each feature in hidden state space.

Run `dca_plots.py` to generate figure S5 (the top row is 2E).

## Figure 3B: Confusion matrices (for RNNs)

Run `confusion_mats.py` to compute and plot the perceptual confusion matrices using existing simulated data (same as the data generated for Fig. 2C using `generate_data_ppss.py`)

## Supplementary figures

### S1C and S2C: Switching behavior

Run `switch_behavior.py` to generate both plots from existing simulated data.

### S2 A and B: Model performance

Run `plot_accuracies.py` to generate the scatterplots of performance measures using the full simulated dataset (A).

Run `generalization_perf_multiseed.py` to create the generalization test plot (B), after generating or downloading simulated data for models in the `correct_choice_model` and `monkey_choice_model` directories. 

### S2D: Psychometric curves

Run `plot_psych_curves.py` to plot the models' psychometric curves as in Fig. S2D. The last cell also plots the monkeys' psychometric curves for comparison.

### S3: Task output activity plots

Run `taskbelief_vs_nNRs.py` to reproduce panel A.  

Run `perc_acc_logreg.py` to reproduce panel B and generate a summary of the logistic regression results. This uses data generated with `generate_data_ppss.py`.

### S6: PCA plots

Run `pca_plots.py` to generate the plots in figure from model weights and trial parameters.

# Model weights

Model weights used in the paper, along with smaller intermediate files to generate figures as described above, can be found in `correct_choice_model/SH2_correctA` and `monkey_choice_model/MM1_monkeyB`.

# Training models and simulating new data

To train new models and generate large datasets for general analysis, see the scripts in `src/train_test_models`. MCM stands for monkey choice model and CCM stands for correct choice model.