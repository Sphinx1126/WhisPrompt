# WhisPrompt
The code and data of "WhisPrompt: Audio Classification of Chinese Opera Genres by Transferring Time-series Features" (under review)

## Datasets
In this paper, we conducted experiments on three fine-grained tasks in total, including:  
+ COG (`./datasets/COG`) is a Chinese Opera Genre Data Set (COG) collected in a pre-experimental stage, which contains 12,000 samples evenly distributed across 6 opera genres.  
+ Due to file size, the divided training set, test set, and validation set can be downloaded [here](https://njuedu-my.sharepoint.cn/:f:/g/personal/522022140097_365_nju_edu_cn/EhOOW6q6GYNCpn6fTJS-hf0BZ2l20Ekmjn4LKR8dmQNtHw?e=aLmt3H).  
+ You can also experiment on your own dataset after splitting it into training set, test set and validation set, and putting the files under the directory `./data/XXX/`. Audio files in various formats (`.mp3`, `.wav`, `.ogg`, etc.) are supported, but the data should be organized as follows:

```
    ─data  
      ├─test  
      │  ├─0  
      │  │  audio-0-0.mp3  
      │  │  audio-0-1.mp3  
      │  │  ...  
      │  ├─1  
      │  │  audio-1-0.mp3  
      │  │  ...  
      │  ├─2  
      │  └─...  
      │  
      ├─train  
      │  └─...  
      │  
      └─valid  
          └─...  
```
## Running
1. Download and install (or update to) the latest release of Whisper with the following command: 
```
pip install -U openai-whisper
```
2. Install the requirements from the `./requirements.txt`
3. Run the `./01_preprocess.py` to convert the audio sample to log-Mel spectrograms and store it as a `.pkl` file.
4. Run the `./02_train.py` to train WhisPAr and evaluate it on the valid set.
5. Run the `./03_test.py` to evaluate WhisPAr on the test set and save the results.
6. You may need to change the hyperparameters if necessary.

## Hyperparameters
You may need to change the hyperparameters in `./01_preprocess.py`, `./02_train.py` and `./03_test.py` for best performance according to your tasks. And here comes the expression of some important hyperparameters.  
+ _data\_dir_: the directory name of your datasets. You should put your data under the directory `./data/data_name/`.  
+ _cls\_num_: the number of category in your datasets.  
+ _audio\_len_: the uniform duration to which an audio file is compressed/padded. Due to the limitations of the pre-trained Whisper, no more than 30s is allowed. Default to 30.  
+ _prompt\_len_: the length of the Prompt sequence added to each Whisper Block. Default to 2.   
+ _bs\_train_ & _bs\_eval_: The sample size contained in a batch when training or evaluating. Default to 8.  
+ _epochs_: Total training epochs.  Default to 100.  
+ _lr_: Preset learning rate. Default to 0.001.  
+ _warmup\_step_: Step of warm-up strategy. Default to 10000.  

## Reference  
+ [https://github.com/openai/whisper](https://github.com/openai/whisper)  
+ [https://www.kaggle.com/datasets/uldisvalainis/audio-emotions](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions/)  
+ [https://www.kaggle.com/competitions/birdclef-2022](https://www.kaggle.com/competitions/birdclef-2022/)  
