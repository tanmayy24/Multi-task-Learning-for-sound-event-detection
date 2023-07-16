### DCASE2022 Task 4 MTL Setup for Sound Event Detection in Domestic Environments.

---

## Requirements
The script `conda_create_environment.sh` is available to create an environment that runs the
following code (recommended to run line by line in case of problems).

## Description
In this study, we leverage some distinctive high-level acoustic characteristics of various sound events to assist the SED model training, without requiring additional labeled data. We use the DCASE Task 4 2022 dataset and categorize the 10 classes into four subcategories based on their high-level acoustic characteristics. We then introduce a novel multi-task learning framework that jointly trains the SED and high-level acoustic characteristics classification tasks using shared layers and weighted loss. Our method significantly improves the performance of the SED system, achieving a 36.3% improvement in terms of the polyphonic sound event detection score compared to the baseline on the DCASE 2022 Task 4 validation set. 

![Screenshot](system.png)


## Dataset
You can download the development dataset using the script: `generate_dcase_task4_2022.py`.
The development dataset is composed of two parts:
- real-world data ([DESED dataset][desed]): this part of the dataset is composed of strong labels, weak labels, unlabeled, and validation data which are coming from [Audioset][audioset].

- synthetically generated data: this part of the dataset is composed of synthetically soundscapes, generated using [Scaper][scaper]. 

### Usage:
Run the command `python generate_dcase_task4_2022.py --basedir="../../data"` to download the dataset (the user can change basedir to the desired data folder.)

If the user already has downloaded part of the dataset, it does not need to re-download the whole set. It is possible to download only part of the full dataset, if needed, using the options:

 - **only_strong** (download only the strong labels of the DESED dataset)
 - **only_real** (download the weak labels, unlabeled and validation data of the DESED dataset)
 - **only_synth** (download only the synthetic part of the dataset)

 For example, if the user already has downloaded the real and synthetic part of the set, it can integrate the dataset with the strong labels of the DESED dataset with the following command:

 `python generate_dcase_task4_2022.py --only_strong` 

 If the user wants to download only the synthetic part of the dataset, it could be done with the following command: 

 `python generate_dcase_task4_2022.py --only_synth`

### Development dataset

The dataset is composed of 4 different splits of training data: 
- Synthetic training set with strong annotations
- Strong labeled training set **(only for the SED Audioset baseline)**
- Weak labeled training set 
- Unlabeled in the domain training set

#### Synthetic training set with strong annotations

This set is composed of **10000** clips generated with the [Scaper][scaper] soundscape synthesis and augmentation library. The clips are generated such that the distribution per event is close to that of the validation set.

The strong annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]`

For example: YOTsn73eqbfc_10.000_20.000.wav 0.163 0.665 Alarm_bell_ringing

#### Strong labeled training set 

This set is composed of **3470** audio clips coming from [Audioset][audioset]. 

**This set is used at training only for the SED Audioset baseline.** 

The strong annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]`

For example: Y07fghylishw_20.000_30.000.wav 0.163 0.665 Dog


#### Weak labeled training set 

This set contains **1578** clips (2244 class occurrences) for which weak annotations have been manually verified for a small subset of the training set. 

The weak annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][event_labels (strings)]`

For example: Y-BJNMHMZDcU_50.000_60.000.wav Alarm_bell_ringing,Dog

#### Unlabeled in the domain training set

This set contains **14412** clips. The clips are selected such that the distribution per class (based on Audioset annotations) is close to the distribution in the labeled set. However, given the uncertainty on Audioset labels, this distribution might not be exactly similar.

The dataset uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] and [desed_real][desed]. 

For more information regarding the dataset, please refer to the [previous year DCASE Challenge website][dcase_21_dataset]. 

## Training
We provide the following **setup** for the task:
- Multi-Task Learning Framework (MTL Framework)

### How to run the system
The **SED system** can be run from scratch using the following command:

`python train_sed_stage_two_MTL.py.py`

---

Note that the default training config will use GPU 0. 
Alternatively, we provide a [pre-trained checkpoint][zenodo_pretrained_models] along with tensorboard logs. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed_stage_two_MTL.py.py --test_from_checkpoint /path/to/downloaded.ckpt`

The tensorboard logs can be tested using the command `tensorboard --logdir="path/to/exp_folder"`. 

**Hyperparameters** can be changed in the YAML file (e.g. lower or higher batch size).

Training can be resumed using the following command:

`python train_sed_stage_two_MTL.py.py --resume_from_checkpoint /path/to/file.ckpt`

#### Results:

| System              | PSDS1 | PSDS2 | PSDS1 + PSDS2 |
|---------------------|-------|-------|---------------|
| Baseline (Audioset)           | 0.351 | 0.552 | 0.903         | 
| Two-stage system (TSS)           | 0.472 | 0.721 | 1.193         | 
| TSS + MTL (α=0.5)                | 0.476 | 0.751 | 1.227         | 
| TSS + MTL (α=0.6)                | 0.457 | 0.740 | 1.197         | 
| TSS + MTL (α=0.7)                | 0.479 | 0.738 | 1.217         | 
| TSS + MTL (α=0.8)                | 0.480 | 0.751 | 1.231         | 
| TSS + MTL (α=0.9)                | 0.490 | 0.729 | 1.219         | 

#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Turpault, Nicolas, et al. "Sound event detection in domestic environments with weakly labeled data and soundscape synthesis."

[3] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[4] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016)

[5] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).

[6] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).

[7] Ronchini, Francesca, et al. "The impact of non-target events in synthetic soundscapes for sound event detection." arXiv preprint arXiv:2109.14061 (DCASE2021)

[8] Ronchini, Francesca, et al. "A benchmark of state-of-the-art sound event detection systems evaluated on synthetic soundscapes." arXiv preprint arXiv:2202.01487 

[9] Khandelwal, T., R. K. Das, et al. “A Multi-Task Learning Framework for Sound Event Detection using High-level Acoustic Characteristics of Sounds”. Interspeech.
