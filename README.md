# An automated way to get insights from face sketches

In a previous job, I had to go through a consequent amount of files to get some information. This meant opening pdf or other formats one after another to go to a specific schema and note what might be noted on it. This is truly time consumming.

As I got more and more used to work with Python, I wanted to develop an algorithm able to automate this task. For this project, I decided to work on extracting insights from face schemas (for instance annotations such as circles and cross) since it is quite general and could be of use to someone reading this.

The algorithm should perform the following tasks:
-   go over the files in a given repository,
-   extract the images in those files,
-   compare them with a model to keep only the information needed,
-   crop, reshape and process the resulting images to analyze and compile the insights,
-   offer an overview of the data,
-   store the results.

## V1:
Works with png files.

## V2:
Works with pdf files.

Here is a sample of the current result:
![GitHub projects](https://user-images.githubusercontent.com/104162893/185877030-e8767580-8e25-44e5-ba1c-cec10eb32f33.jpg)

**Potential issues anticipated so far**:
- A bad scan prevent a good cropping. In those cases it is detected by the program.
- A blank page or schema without annotations is anticipated and also reported in the csv file.
- Rotated images are skipped by the program, a technique to deal with it is currently tested.

**How to use the code**:
- Before executing the code you should:
    - Put your input images in the v1 folder as png.
    - Have your model image in the Reference folder.
    - Have an empty Analysis_report folder.
    - Change the parameters to adapt it to your files (i.e: the CV2 Houghlines and Houghcircles functions parameters, the spaces between duplicates) --> see in post scriptum.
    - Make sure to be in the Face_report repository before running the script in your terminal.
- What you get after executing the code:
    - A csv file (cross_circles_coord.csv) with each file id with its original name from v1 folder, the coordinates of each annotation an image has and its type, and a message explaining if nothing was extracted from an image.
    - An image (Compilation.jpeg) with the visual compilation of all cross on one face and all circles on another.
    - The execution time in the terminal.

**PS:** 
The parameters to change depending on your files will be accessible in one place in the next update.
The program is still under development. I intend to make it more efficient, cleaner and add a sharing feature.
