# DDKtor
system for detection of VOT and Vowels for ddk task


Yael Segal(segal.yael@campus.technion.ac.il)\
Joseph Keshet (jkeshet@technion.ac.il)             

             
DDKtor is a software package for automatic measurement of voice onset time (VOT) and vowels.
We propose a neural-based architecture composed of CNN as fearues selector and a recurrent neural network(RNN) for classification. DDKtor can handle VOT that appears in a middle of a word and short vowels from ddk task. 

This is a beta version of DDKtor. Any reports of bugs, comments on how to improve the software or documentation, or questions are greatly appreciated, and should be sent to the authors at the addresses given above.


## Installation instructions

- Python 3.7+

- Download the code:
    ```
    git clone https://github.com/YaelSegal/DDKtor
    ```
- Download Praat from: http://www.fon.hum.uva.nl/praat/ .

- Download SoX from: http://sox.sourceforge.net/ .

- To verify everything is installed, change dir to DDKtor, and run the ```check_installations.sh``` script:
    ```
    $ ./check_installations.sh
     ```
  Note: maybe you will need to change the execution permissions as: ```$ chmod +x ./check_installations.sh```
  
  
If you encounter any problem, please check the ```log.txt```.

## How to use:

- Place your ```.wav``` files in the ```./data/raw/ ``` directory.  \
Note:You can also place directories that contain the ```.wav``` files, the is no need to re-arrange your data. For example:
    ```
    ./data/raw
            └───dir1
            │   │   1.wav
            │   │   2.wav
            │   │   3.wav
            │               │   
            └───dir2
                │   1.wav
                │   2.wav
                │   3.wav
    ```

- You can run one of the following scripts: run_script.sh

Run the following script:

### run_script.sh:

This script uses pairs of textgrids and wavs (must have the same name). Each textgrid must contain a window tier that define where the system should look for VOTs and vowels(this tier can contain more than one window).
This script runs DDKtor on the windows.
There are few optional ways to run the script. The first one is without any parameter so the name of the windows tier will have to be "window". The second one is with parameter - where the parameter is the name of the winodws tier. 
In addition, there is an option to run the model with pre-processing to reduce noise (using the package noisereduce)
    
    ```
    $ ./run_script.sh # by default the window tier is called window, without noise reduction
    $ ./run_script.sh -w area #  the window tier is called area, without noise reduction
    $ ./run_script.sh -n # the window tier is called  window, with noise reduction
    $ ./run_script.sh -w area -n #  the window tier is called area, with noise reduction
    ```         

    Note: make sure it has an execute permission. (```$ chmod +x ./run_script.sh```)


- The predictions can be found at ```./data/out_tg/ ``` in the same hierarchy as the original data.
For example:

```
    ./data/out_tg
    | 
    └───dir1
    │   │   1.TextGrid
    │   │   2.TextGrid
    │   │   3.TextGrid
    │               
    └───dir2
        │   1.TextGrid
        │   2.TextGrid
        │   3.TextGrid
```
  
