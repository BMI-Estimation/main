# Digital Estimation of Body Mass Index

The following code repository contains the source code for the ELEN4002 Laboratory Project.
The ELEN4002 course is to be completed for the degree of Bachelor of Sciences in Engineering (Electrical), for the School of Electrical and Information Engineering at the University of the Witwatersrand, Johannesburg.
The purpose of this project is to determine, to a reasonable degree of certainty, the Body Mass Index (BMI) of a person from their photograph.

BMI is defined by the following ratio:
BMI = mass/(height^2)

To use this program, you need two photographs of a person facing forwards and to the side.
In these photographs, there should be a black rectangular object in both photographs, on the left hand side of each photograph.
Two samples are provided in 'images', shown below.

![Front Example](images/C001_F.jpg)
![Side Example](images/C001_S.jpg)

As a safeguard for front and side images accidentally being swopped, an image is considered as a front image if it has the letter 'f' in it's name, and a side image if it has the letter 's' in it's name.

---

## Requirements

This project was done using Python 3.7.4, however it should work with any Python 3 installation.
Should there be required packages not supported by the current distribution of Python, please use Python 3.7.4 as available from [python.org](https://www.python.org/downloads/).

Please ensure that the default python package installer *pip* has been updated by running the following:

    pip install --upgrade pip

---

## Installation Instructions

Clone this repository using Git Bash, Windows Powershell or your preferred terminal.

    git clone git@github.com:djsing/ELEN4002LabProject.git

Alternatively, if you do not have git installed on your machine, download the zip file from this page by clicking 'Clone or download', followed by 'Download Zip'.

Open the terminal in the root directory of the repository and run the following command:

    pip install -r requirements.txt

---

## Data Generation Instructions

To generate the front and side view data from your image dataset, first copy your images into the 'images' folder.
Please ensure the files are named correctly as described above, and that there are a equal number of front and side photographs in the folder.

Download the [Mask-RCNN](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) trained on the COCO Dataset, place it in root directory.

Next, run the BMI_Estimation.py file from the terminal.

The following additional arguments are required:

- -w , followed by the known width of the reference object in meters

The following optional arguments are also allowed:

- -m , gives the option to view the pixel masks created during the program execution

- -v , provides a visualisation of where the pixels mask originate from on the original photographs

For example, if you wish to generate data without seeing either the visualisation or the pixel masks, you would run the following command:

    python BMI_Estimation.py -w 0.29

If you wish to see each pixel mask but not the original photographs:

    python BMI_Estimation.py -w 0.29 -m

If you wish to see both the pixel masks and the original photographs:

    python BMI_Estimation.py -w 0.29 -m -v

> *Note: The order of these additional arguments does not matter, however arguments which require succeeding values must follow the order *-arg value*, eg: -w 0.29

After training, 'front.csv' will consist of dimensions estimated using the front photograph, while 'side.csv' will contain those estimated from the side photograph.
Both .csv files will contain a row per photograph,with the data in the following column order:

1. 2D Area
2. Height
3. Horizontal Width - Thorax
4. Horizontal Width - Abdomen
5. Horizontal Width - Pelvis
6. Horizontal Width - Upper Leg (thigh)

> *Note: The images in the folder will be read in alphabetical/numerical order. It is advised that the photographs be labelled in the same way.

---

## Training Front and Side Models

Ensure that your 'front.csv' and 'side.csv' files containing all the front and side estimates are still in the root directory.
Next, the output data file 'BMI.csv' should also be placed in the root directory.
Each row in the 'BMI.csv' file should correspond to the same row in the 'front.csv' and/or 'side.csv' files (depending on whether you wish to train the Front or Side models) at the time of training.
The column order of the 'BMI.csv' file is as follows:

1. Mass (kg)
2. Height (m)
3. BMI

Next, run the train_image_model.py file from the terminal.

The following additional arguments are required:

- -b , followed by the desired batch size for training. Must be greater than 0.

- -e , followed by the desired number of epochs per training cycle. Must be greater than 0.

- n , followed by the desired number of training cycles. Must be greater than 0.

- -fo , followed by the desired number of folds for Cross Validation.

**Only** one of the two following two arguments must also be provided:

- -f , indicating that the Front model is to be trained. This will use the data in 'front.csv'.

- -s , indicating that the Side model is to be trained. This will use the data in 'side.csv'.

Optional Arguments:

- -st , indicating that training should use Stratified Cross Validation instead of standard Cross Validation.

- -v , indicating that you would like to view the results of training.
This includes model loss, and response to both testing and unseen data.

### Extra Features

**One** of following additional arguments may also be provided:

- -he , which will create a model that predicts height from both the front and side data.

> *Note: For this mode, both 'front.csv' and 'side.csv' must be present, with each row in both corresponding to the same person.
Also note that using this argument will override both '-f' and '-s' arguments.

- -m , which will create a model that predicts mass instead of BMI. In conjunction with the creation of a height model, this provides another means of calculating BMI.

Models with the best performance are automatically saved in models/*Date_and_Time_of_Running*/BMI/*Network_Architecture_Used*/Best_*Classical_or_Cross*_*Front_or_Side*.h5

> *Note: When using the script in Height or Mass mode, it will still use the same directory structure, but will have a folder named 'Height' or 'Mass' instead of 'BMI'.

All other .h5 files produced and saved in the root directory are files left over from training and can be discarded or remain in the directory as they are overwritten each training iteration.

Using the training performance metrics saved alongside the best performing models, you have the freedom to choose which model is appropriate for you.

---

## Training the Compensation Model

Ensure that your 'front.csv' and 'side.csv' files containing all the front and side estimates are still in the root directory.
Also ensure that both the desired Front and Side models that you created in the previous section are now copied into the root directory, renamed as 'Front.h5' and 'Side.h5'.

Next, run the train_bmi_model.py from the terminal.

This script shares the following arguments with the script in the 'Training Front and Side Models' section, with the same usage.

- -b
- -e
- -n
- -fo
- -st
- -v

Models with the best performance are saved using the same directory structure as mentioned in the previous 'Training Front and Side Models' section, but will have a folder named 'BMI_Comp' instead of 'BMI'.
The best models will be saved as 'Best_*Classic_or_Cross*.h5' in this folder with its training performance.

All other .h5 files produced and saved in the root directory are files left over from training and can be discarded or remain in the directory as they are overwritten each training iteration.

---

## Using the GUI for BMI Prediction

The user interface is shown below.

![GUI](assets/gui.JPG)

The steps to using the interface are as follows:

1. Enter the width (horizontal) of the reference square.

2. Click 'Browse Files' for 'Front Image', choose your front image. Ensure that the filename contains an 'f' or 'F', but no 'S' or 's'.

3. Click 'Browse Files' for 'Side Image', choose your front image. Ensure that the filename contains an 's' or 'S', but no 'f' or 'F'.

Your interface should now resemble this:

![Used GUI](assets/usedgui.JPG)

After selecting the photographs, you have the following options:

- 'Only Use Front Image':
  - If you only have a front image selected, tick this option.

- 'Only Use Side Image':
  - If you only have a side image selected, tick this option.

If you attempt to click 'Predict BMI' after only selecting one photograph and not selecting the relevant option above, the program will raise an error.

- 'Show Masks'
  - Choose this option to show the masks of the reference object and person detection.
  The program pauses its execution while any of the image windows are still open.
  
  - Choose this option if you need to see if there were any errors during the image processing.

- 'Show Image Segmentation'
  - Shows the separation of the pixel masks from the original image.
  
  - Shows the pixel masks from foreground extraction as well as the pixel masks extracted from the Mask RCNN object detection.

## Evaluating Model Performance

Place the following files in to the root directory:

- front.csv
- side.csv
- Front.h5
- Side.h5
- BMI.h5
- BMI.csv

Run the Analysis.py file.

    python Analysis.py

> *Note: Later releases will include the ability to only analyse Front, Side and BMI models separately.
For the moment, all of the above files are required.
If you have not trained all the models or have all the data as described earlier in these instructions, use the default files as provided in this repo.
