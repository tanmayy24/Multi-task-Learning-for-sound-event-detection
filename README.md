# A Multi-Task Learning Framework for Sound Event Detection using High-level Acoustic Characteristics of Sounds
Domestic environment sound event detection task leveraging some distinctive high-level acoustic characteristics of various sound events to assist the SED model training, without requiring additional labeled data.
---

## DCASE Task 4
Multitask Learning DCASE Task 4 recipe: 
- [DCASE 2022 Task 4](./recipes/dcase2022_task4_baseline)

Challenge website [here][dcase_website] 

[dcase_website]: https://dcase.community
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[invite_dcase_slack]: https://join.slack.com/t/dcase/shared_invite/zt-mzxct5n9-ZltMPjtAxQTSt3a6LFIVPA
[slack_channel]: https://dcase.slack.com/archives/C01NR59KAS3

## Installation Notes

### You want to run the MTL DCASE 2022 Task 4 system

Go to `./recipes/dcase2022_task4_baseline` and follow the instructions there in the `README.md`

In the recipe, we provide a conda script that creates a suitable conda environment with all dependencies, including 
**pytorch** with GPU support in order to run the recipe. There are also instructions for data download and preparation. 


### You need only desed_task package for other reasons
Run `python setup.py install` to install the desed_task package 


## Citation
**If this work is helpful, please feel free to cite the following paper:**

Khandelwal, T. and R. K. Das. 2023. “A Multi-Task Learning Framework for Sound Event Detection using High-level Acoustic Characteristics of Sounds”. Interspeech. <br />
To access the paper<br />
[Arxiv](https://arxiv.org/abs/2305.10729)





