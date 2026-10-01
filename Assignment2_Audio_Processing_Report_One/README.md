# Audio Processing Report One

## Title

**Time–Frequency Analysis of Environmental Sounds Using MATLAB**

## Selected Sounds

Three representative five-second recordings from the official ESC-50 dataset are now fixed for the report:

- Rain — `audio/rain.wav` (source: `1-17367-A-10.wav`)
- Dog bark — `audio/dog_bark.wav` (source: `1-30226-A-0.wav`)
- Clock tick — `audio/clock_tick.wav` (source: `1-21934-A-38.wav`)

All three source files belong to ESC-50 fold 1 and are ESC-10 categories.

## Aim

The report investigates how time-domain waveforms, Fourier spectra, and Short-Time Fourier Transform (STFT) spectrograms reveal different characteristics of sustained, non-stationary, and transient environmental sounds.

A second focus examines how STFT window length changes the time–frequency representation, especially for the transient clock-tick signal. The report includes both visual comparisons and a quantitative experiment using a fixed FFT size and hop size.

## Completed Processing Pipeline

1. Load and validate the three audio samples.
2. Convert stereo input to mono if required.
3. Inspect sampling rate, duration, and waveform amplitude.
4. Compare time-domain waveforms.
5. Compare FFT magnitude spectra.
6. Compute baseline STFT spectrograms.
7. Compare short, medium, and long STFT windows.
8. Quantify clock-tick time spreading and spectral peak bandwidth using a fixed FFT size.
9. Interpret and discuss the observed time–frequency behaviour.

## Baseline STFT Parameters

- Window: periodic Hann
- Window length: 2048 samples
- Hop size: 512 samples
- FFT size: 2048 samples

The quantitative window-length experiment separately uses window lengths of 512, 2048 and 4096 samples with a fixed FFT size of 4096 samples and a fixed hop size of 128 samples.

## Included Files

- `Report1 HaoliangXiong.mlx` — MATLAB Live Script and primary report source
- `Report1 HaoliangXiong.pdf` — exported PDF report
- `audio/` — the three selected audio samples required to reproduce the report
- `scripts/processing_pipeline.m` — supporting MATLAB processing script
- `figures/processing_pipeline.png` — processing workflow figure
- `functions/` — location reserved for report-specific helper functions
- `results/` — location reserved for small exported numerical results

## Current Report 1 Progress

- [x] Topic fixed
- [x] Repository structure created
- [x] Three ESC-50 samples selected
- [x] Three WAV files added to `audio/`
- [x] Audio provenance documented
- [x] `Report1 HaoliangXiong.mlx` completed and included
- [x] All three samples loaded and validated in MATLAB
- [x] Time-domain comparison completed
- [x] FFT comparison completed
- [x] Baseline STFT comparison completed
- [x] Visual STFT window-length comparison completed
- [x] Quantitative fixed-NFFT window-length experiment completed
- [x] Discussion and conclusion completed
- [x] `Report1 HaoliangXiong.pdf` exported and included
- [x] Final ZIP extracted and verified with a clean MATLAB Run All

## Reproducibility

The report includes the Live Script, exported PDF, three selected WAV files and supporting files required for reproduction. The complete ESC-50 dataset is not included because only the three selected Report 1 samples are required.

**Final ZIP verification status:** Pending. This status must be changed to verified only after the final ZIP has been created, extracted into a clean directory and successfully executed using MATLAB Run All with only the extracted contents.
