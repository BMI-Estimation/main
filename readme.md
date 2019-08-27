# Digital Estimation of Body Mass Index

The following code repository contains the source code for the ELEN4002 Laboratory Project.
The purpose of this project is to determine, to a reasonable degree of certainty, the Body Mass Index (BMI) of a person from their photograph.

To use this program, you need two photographs of a person facing forwards and to the side.
In these photographs, there should be a black rectangular object in both photographs, on the left hand side of each photograph.
Two samples are provided in 'images', shown below.

![Front Example](images/C001_F.jpg)
![Side Example](images/C001_S.jpg)

As a safeguard for front and side images accidentally being swopped, an image is considered as a front image if it has the letter 'f' in it's name, and a side image if it has the letter 's' in it's name.

---

## Requirements

This project was done using Python 3.7.4, however it should work with any Python 3 installation.
Should there be required packages not supported by the current distrbution of Python, please use Python 3.7.4 as available from [python.org](https://www.python.org/downloads/).

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

Next, run the train_image_model.py file from the terminal.

The following additional arguments are required:

- -b , followed by the desired batch size for training. Must be greater than 0.

- -e , followed by the desired number of epochs per training cycle. Must be greater than 0.

- n , followed by the desired number of training cycles. Must be greater than 0.

**Only** one of the two following two arguments must also be provided:

- -f , indicating that the Front model is to be trained. This will use the data in 'front.csv'.

- -s , indicating that the Side model is to be trained. This will use the data in 'side.csv'.

