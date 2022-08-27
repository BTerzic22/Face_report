import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import color, measure
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage import morphology
import glob, cv2, os, time


def df_creator():
    """Creates the dataframe to store analysis results
    Returns the dataframe
    """
    coord_df = pd.DataFrame()
    coord_df[['file_id', 'file_name', 'tuple_coord', 'coord_x', 'coord_y', 'type']] = 0
    return coord_df


def mask_dilution(im_cal):
    """Modifies the reference image to create a mask which will be used on images to analyse
    Returns the mask
    """
    # make a copy of ref image
    dilated_mask = im_cal.copy()
    lim_left, lim_right = round(im_cal.shape[1]*1/4), round(im_cal.shape[1]*3/4)
    dil_left = morphology.erosion(im_cal[:, :lim_left], morphology.disk(12))
    dil_right = morphology.erosion(im_cal[:, lim_right:], morphology.disk(12))
    dil_inner = morphology.erosion(im_cal[:, lim_left:lim_right], morphology.disk(8))
    # update the mask with the eroded areas
    dilated_mask[:, lim_right:] = dil_right
    dilated_mask[:, :lim_left] = dil_left
    dilated_mask[:, lim_left:lim_right] = dil_inner
    # convert to numpy array
    dilated_mask = np.array(dilated_mask)*1
    return dilated_mask


def initialize():
    """Initializes the variables needed for the program
    Returns those variables
    """
    im_cal = calibration()
    coord_df = df_creator()
    file_id, images_to_test = 0, list()
    path = os.getcwd()
    local_im_list = glob.glob(path + "/Face_report/v1/*.PNG")
    dilated_mask = mask_dilution(im_cal)
    return im_cal, coord_df, file_id, images_to_test, local_im_list, dilated_mask


def bordure_detection(face, cal):
    """Detects the vertical lines in the image which will be used later for cropping. If nothing
     is detected, an error value will be returned.
    Returns the coordinates and the error statement
    """
  # Convert image to grayscale
    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
  # Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)
  # Apply HoughLinesP method to directly obtain line end points
    lines = cv2.HoughLinesP(
              edges, # Input edge image
              1.5, # Distance resolution in pixels
              np.pi/180, # Angle resolution in radians
              threshold=40, # Min number of votes for valid line
              minLineLength=40, # Min allowed length of line
              maxLineGap=15 # Max allowed gap between line for joining them
              )
    
    X, Y = list(), list()
  # Iterate over points
    if type(lines)== None:
        return X, Y, 1
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Get margins
        # Select only vertical lines for non ref image
        if cal==0:
          if y1 != y2 and np.abs(x1-x2) <= 2:
            X.append(x1); X.append(x2)
            Y.append(y1); Y.append(y2)
        # Select only vertical lines for ref image
        elif np.abs(x1-x2) < 5 :
          # Maintain a simples lookup list for points
          X.append(x1); X.append(x2)
          Y.append(y1); Y.append(y2)
  # Save the result image
    if len(X)==0 and len(Y)==0:
        error = 1
    else:
        error = 0
    return X, Y, error


def im_croping(X, Y, img):
    """Crops the image based on the min and max coordinates of the vertical lines which are the
     cadran of the face sketch, and then adds margins.
    Returns the newly cropped image
    """
    left = np.min(X)
    top = np.min(Y)
    right = np.max(X)
    bottom = np.max(Y)
    delta = 10
    img_res = img.crop((left - delta, top - delta, right + delta, bottom + delta)) 
    return img_res


def calibration():
    """Uses the reference image as a caliber for analyzed images 
    Returns the caliber image
    """
    path = os.getcwd()
    ref_path = path + "/Face_report/Reference/Reference.PNG"
    sample = Image.open(ref_path)
    np_sample = np.array(sample)
    X_cal, Y_cal, error = bordure_detection(np_sample, cal=1)
    if error == 1:
        print('There is an issue with the reference image.')
    im_with_face = im_croping(X_cal, Y_cal, sample)
    im_test_rgb = im_with_face.convert('RGB')

    # Convert into grayscale image
    im_test_gray = color.rgb2gray(im_test_rgb)
    thresh = threshold_otsu(im_test_gray)
    im_cal = im_test_gray > thresh
    return im_cal


def check_annotation(contour, cross_dots):
    """Selects the mean coordinates of an annotation. Checks if they are out of margins area.
    Returns the check validation and the center coordinates
    """
    check, margins = False, 25
    col_center = np.mean(contour[:, 1], dtype=int)
    row_center = np.mean(contour[:, 0], dtype=int)
    # eliminate contour within the margins on image sides
    if row_center < margins or row_center > cross_dots.shape[0]-margins:
        return check, row_center, col_center
    elif col_center < margins or col_center > cross_dots.shape[1]-margins:
        return check, row_center, col_center
    else:
        check = True

    # This section is in case of you would like to focus only on a particular area of the face:
        # selection of a plausible area to make sur to not select a small stain as a contour
        #if row_center > cross_dots.shape[0] * 2/5:
        #    check = True
        #elif col_center > cross_dots.shape[1] * 2/5 and col_center < cross_dots.shape[1] * 3/5:
        #    check = True

    return check, row_center, col_center


def duplicates_finder(contours_storage):
    ''' Eliminates a value when it is spacially too close to another. This is to prevent us from having
    two annotations where the algorithm should only have detected one.

    Returns the updated contours_storage list.
    '''
    del_list = list()
    # remove identic values
    contours_storage = list(set(contours_storage))
    # iterate through the list to compare it to itself
    for center in contours_storage:
        i = 0
        # the range is set to 40 pixels but it may be increased if needed
        while i in range(len(contours_storage)):
            if center == contours_storage[i] or center in del_list:
                i+=1
                continue
            elif center[0] >= (contours_storage[i][0] - 12) and center[0] <= (contours_storage[i][0] + 12):
                if center[1] >= (contours_storage[i][1] - 12) and center[1] <= (contours_storage[i][1] + 12):
                    del_list.append(contours_storage[i])
            i+=1
        contours_storage = [var for var in contours_storage if var not in del_list]
    return contours_storage


def ellipsis_filter(centers, im_thresh):
    """Checks if the annotation is within an ellipsis shape fitting the actual face shape.
    Returns the centers within the ellipsis shape
    """
    a = round(np.shape(im_thresh)[0]/2)
    b = round(np.shape(im_thresh)[1]/2)
    def y(x):
        y = np.sqrt(a**2*(1 - (x-b)**2/b**2))+a
        return y
    def z(x):
        z = - np.sqrt(a**2*(1 - (x-b)**2/b**2))+a
        return z

    keep_list = list()
    for c in centers:
        if c[0] > z(c[1]) and c[0] < y(c[1]):
            keep_list.append(c)
    centers = [c for c in keep_list]
    return centers


def image_threshold(img_res, im_cal):
    """Takes the cropped image, converts to RGB (useful for PNG with a 4th channel), then to
     grayscale, puts it the the caliber image size and uses a threshold
    Returns the binary image
    """
    im_test_rgb = img_res.convert('RGB')
    im_test_gray = color.rgb2gray(im_test_rgb)
    im_test_gray = resize(im_test_gray, (np.shape(im_cal)[0], np.shape(im_cal)[1]))
    thresh = threshold_otsu(im_test_gray)
    im_thresh = im_test_gray > thresh
    return im_thresh


def cross_cricles(im_thresh, dilated_mask):
    """Uses the mask from reference image to get rid of everything but the annotation
    Returns the 'just annotation' image
    """
    cross_dots = dilated_mask - np.array(im_thresh)*1
    cross_dots = np.where((cross_dots==-1), 0, cross_dots)
    cross_dots = np.where((cross_dots==0)|(cross_dots==1), cross_dots^1, cross_dots)
    cross_dots = cross_dots*255
    cross_dots = np.uint8(np.around(cross_dots))
    return cross_dots


def store_contours(cross_dots):
    """Identifies all the contours in the image
    Returns the list of contours
    """
    cs = list()
    contours = measure.find_contours(cross_dots)
    for contour in contours:
        check, row_center, col_center = check_annotation(contour, cross_dots)
        if check == True:
            cs.append((row_center, col_center))
    contours_storage = duplicates_finder(cs)
    return contours_storage


def circles_detector(cross_dots):
    """Detects and stores the circle shape annotations in the image.
    Returns circle(s) center(s) list
    """
    # finds the circles in the grayscale image using the Hough transform
    circles = cv2.HoughCircles(image=cross_dots, method=cv2.HOUGH_GRADIENT, dp=2, 
                                minDist=20, param1=300, param2=10, minRadius=2, maxRadius=30)
    circle_centers=list()

    if circles is not None:
        # Convert the circle parameters a, b and r to integers.
        circles = np.uint16(np.around(circles))

        for pt in circles[0, :]:
            a, b = pt[0], pt[1]
            if cross_dots[b, a] != 255:
                circle_centers.append((b,a))
    # find duplicates
    circle_centers = duplicates_finder(circle_centers)
    return circle_centers


def sep_cross(contours_storage, circle_centers, im_thresh):
    """Compares the all annotation and circles lists to keep only non-circles centers from the
     1st list. Allocates those centers to the cross(es) center(s) list.
    Returns the cross(es) center(s) list
    """
    cross_centers = contours_storage.copy()

    for i in range(len(cross_centers)):
        for j in range(len(circle_centers)):
            if (cross_centers[i][0] >= circle_centers[j][0]-10) and (cross_centers[i][1] >= circle_centers[j][1]-10) and (cross_centers[i][0] <= circle_centers[j][0]+10) and (cross_centers[i][1] <= circle_centers[j][1]+10):
                cross_centers[i] = (0,0)
    cross_centers = [i for i in cross_centers if i != (0, 0)]
    cross_centers = ellipsis_filter(cross_centers, im_thresh)
    circle_centers = ellipsis_filter(circle_centers, im_thresh)
    print('cross:', len(cross_centers),'\n','circles:', len(circle_centers))

    return cross_centers


def round_coord(centers):
    """Rounds the coordinates because we do not need to be that precise. It is also interesting
     for comparing to manual analysis.
    Returns rounded coordinates
    """
    centers = [(int(np.round(c[0], -1)), int(np.round(c[1], -1))) for c in centers]
    return centers

def update_df(coord_df, circle_centers, cross_centers, file_id, local_im):
    """Updates the results dataframe with the image results
    Returns the updated dataframe
    """
    # initialize the two dataframes
    df_o, df_x = pd.DataFrame(), pd.DataFrame()
    # round the coordinates
    circle_centers = round_coord(circle_centers)
    cross_centers = round_coord(cross_centers)
    # fill the circle dataframe
    df_o['tuple_coord'] = circle_centers
    df_o['coord_x'] = [c[0] for c in circle_centers]
    df_o['coord_y'] = [c[1] for c in circle_centers]
    df_o['type'] = 'circle'
    df_o['file_id'] = file_id
    df_o['file_name'] = local_im
    # fill the cross dataframe
    df_x['tuple_coord'] = cross_centers
    df_x['coord_x'] = [c[0] for c in cross_centers]
    df_x['coord_y'] = [c[1] for c in cross_centers]
    df_x['type'] = 'cross'
    df_x['file_id'] = file_id
    df_x['file_name'] = local_im
    # concatenate with the global dataframe to update it
    coord_df = pd.concat([coord_df, df_o, df_x])
    return coord_df


def non_nominal_update_df(coord_df, message, file_id, local_im):
    """Updates the results dataframe with information on the non-nominal result for this image
    Returns the updated dataframe
    """
    # update the global dataframe with non nominal case report
    df = pd.DataFrame(columns=['tuple_coord', 'coord_x', 'coord_y', 'type', 'file_id', 'file_name'])
    df.loc[0] = [np.nan, np.nan, np.nan, message, file_id, local_im]
    coord_df = pd.concat([coord_df, df])
    return coord_df


def pd_to_csv(coord_df):
    """Transforms the pandas dataframe to a csv file and saves it.
    Returns nothing
    """
    path = os.getcwd()
    coord_df.to_csv(path+'/Face_report/cross_circles_coord.csv', index=False)


def verification_by_image(images_to_test, coord_df):
    """Stores the algorithm analysis for each image as a plot
    Returns nothing
    """
    id=0
    for im in images_to_test:
        cimg = im.copy()
        cimg = np.uint16(cimg)
        coord = coord_df[coord_df['file_id']==id]
        circles = coord[coord['type']=='circle']
        if len(circles) != 0:
            for contour in circles['tuple_coord']:
               cv2.circle(img=cimg,center=(contour[1], contour[0]),radius=45,color=(0,255,0),thickness=2)
        cross = coord[coord['type']=='cross']
        if len(cross) != 0:
            for contour in cross['tuple_coord']:
                cv2.circle(img=cimg,center=(contour[1], contour[0]),radius=20,color=(0,0,255),thickness=2)
        plt.title(coord_df['file_name'].unique()[id])
        plt.imshow(cimg, cmap=plt.cm.gray)
        path = os.getcwd()
        plt.savefig(path+"/Face_report/Analysis_report/{}.jpeg".format(id))
        plt.clf()
        id+=1


def compil_cross_and_circles(coord_df, im_cal):
    """Plots and save the compilation of all circles and cross coordinates
    Returns nothing
    """
    a = 0.5; s = 100
    data1 = coord_df[coord_df['type']=='circle']
    data2 = coord_df[coord_df['type']=='cross']

    fig, ax = plt.subplots(1,2, sharey = True, sharex = True, figsize=[20, 15])
    plt.suptitle("Compilation of annotations")

    ax[0].imshow(im_cal, cmap=plt.cm.gray)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_title('Circles')
    # warning: x and y axis are inverted in the reference image
    sns.scatterplot(data = data1 , y = 'coord_x', x = 'coord_y', color = 'r', alpha=a, s=s, ax=ax[0])

    ax[1].imshow(im_cal, cmap=plt.cm.gray)
    ax[1].set_title('Crosses')
    sns.scatterplot(data = data2 , y = 'coord_x', x = 'coord_y', color = 'b', alpha=a, s=s, ax=ax[1])
    sns.despine()

    path = os.getcwd()
    plt.savefig(path+"/Face_report/Compliation.jpeg")
    plt.clf()


def main():
    # Initialize
    im_cal, coord_df, file_id, images_to_test, local_im_list, dilated_mask = initialize()
    # iterate through the pdf list
    for local_im in local_im_list:
        print('-------------- ', 'image {} over {}'.format(file_id+1, len(local_im_list)), ' --------------')
        print(local_im)
        im = Image.open(local_im).convert('RGB')
        np_im = np.array(im)

        # crop the image
        X, Y, error = bordure_detection(np_im, cal=0)
        if error == 1:
            print("error in face detection!")
            message='Did not manage to detect a face'
            coord_df = non_nominal_update_df(coord_df, message, file_id, local_im)
            images_to_test.append(im_thresh)
            file_id+=1
            continue
        img_res = im_croping(X, Y, im)

        # prepare the image and get the annotations
        im_thresh = image_threshold(img_res, im_cal)
        cross_dots = cross_cricles(im_thresh, dilated_mask)
        #plt.imshow(cross_dots, cmap=plt.cm.gray)
        #plt.show()
        # no need to find contours if there is no mark
        if np.mean(cross_dots)==255:
            print("No marks detected!")
            message='No marks'
            coord_df = non_nominal_update_df(coord_df, message, file_id, local_im)
            images_to_test.append(im_thresh)
            file_id+=1
            continue

        contours_storage = store_contours(cross_dots)
        # if there are too many marks we assume there is a problem reading the file
        if len(contours_storage) > 10:
            print("There is probably an issue processing this image. We skip it for safety.")
            message='Too much marks'
            coord_df = non_nominal_update_df(coord_df, message, file_id, local_im)
            images_to_test.append(im_thresh)
            file_id+=1
            continue

        # find the circles
        circle_centers = circles_detector(cross_dots)

        # determine the cross as non-circle annotations
        cross_centers = sep_cross(contours_storage, circle_centers, im_thresh)

        # get the coordinates in a pandas dataframe
        if len(circle_centers)==0 and len(cross_centers)==0:
            print("No marks detected!")
            message='No marks'
            images_to_test.append(im_thresh)
            coord_df = non_nominal_update_df(coord_df, message, file_id, local_im)
        else:
            coord_df = update_df(coord_df, circle_centers, cross_centers, file_id, local_im)
        images_to_test.append(im_thresh)
        file_id+=1
        
    # save the whole dataframe as an excel file
    pd_to_csv(coord_df)

    # verify the results
    print('Plotting results in progress')
    verification_by_image(images_to_test, coord_df)
    compil_cross_and_circles(coord_df, im_cal)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    m = (end-start)//60
    s = np.round(end-start - m*60, 0) 
    print("Execution time: {} min {} sec".format(m, s))
