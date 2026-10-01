# A Comparative Study of Audio Feature Representations for Environmental Sound Classification

## Research Question

How do MFCC and Log-Mel feature representations compare in accuracy, macro-F1 and computational cost on ESC-50 when evaluated using the same RBF-SVM pipeline?

## Project Overview

This ELEC5305 research project investigates how different audio feature representations influence environmental sound classification performance. The project will compare commonly used representations under a controlled classification pipeline to examine their effectiveness and computational requirements.

## Dataset

The project will use the ESC-50 dataset, a publicly available benchmark containing 2,000 five-second environmental audio recordings across 50 sound classes.

Official dataset repository:

https://github.com/karolpiczak/ESC-50

The dataset will not be stored in this repository. Instructions for downloading and preparing the dataset will be provided separately.

## Proposed Feature Representations

The initial comparison will include:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Log-Mel spectrogram features
- Handcrafted spectral features, if time permits

## Proposed Methodology

The audio recordings will be processed using a consistent preprocessing pipeline. Each feature representation will be evaluated using the same classification method to reduce the influence of unrelated model differences.

The initial classifier will be a Support Vector Machine with a radial basis function kernel. Evaluation will follow the official five-fold structure provided by ESC-50.

## Evaluation Metrics

The proposed evaluation metrics include:

- Classification accuracy
- Macro-F1 score
- Confusion matrix
- Feature extraction time
- Model training and inference time

## Tools

The project is expected to use:

- Python
- Jupyter Notebook
- Librosa
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn

## Current Progress

- [x] Project topic selected
- [x] Research question defined
- [x] Public GitHub repository created
- [x] Initial dataset selected
- [x] Literature review
- [x] Project proposal
- [ ] Feature extraction implementation
- [ ] Classification experiments
- [ ] Evaluation and final report

## Reference
[1] K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," in Proceedings of the 23rd ACM International Conference on Multimedia, pp. 1015-1018, 2015. DOI: 10.1145/2733373.2806390.

[2] S. Chu, S. Narayanan, and C.-C. J. Kuo, "Environmental Sound Recognition With Time-Frequency Audio Features," IEEE Transactions on Audio, Speech, and Language Processing, vol. 17, no. 6, pp. 1142-1158, 2009. DOI: 10.1109/TASL.2009.2017438.

[3] K. J. Piczak, "Environmental Sound Classification with Convolutional Neural Networks," in 2015 IEEE 25th International Workshop on Machine Learning for Signal Processing, pp. 1-6, 2015. DOI: 10.1109/MLSP.2015.7324337.

[4] M. K. Gourisaria, R. Agrawal, M. Sahni, and P. K. Singh, "Comparative Analysis of Audio Classification with MFCC and STFT Features Using Machine Learning Techniques," Discover Internet of Things, vol. 4, art. 1, 2024. DOI: 10.1007/s43926-023-00049-y.

[5] S. Chachada and C.-C. J. Kuo, "Environmental sound recognition: a survey," APSIPA Transactions on Signal and Information Processing, vol. 3, e14, 2014. DOI: 10.1017/ATSIP.2014.12.
