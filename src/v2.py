import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import color, measure
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage import morphology
import glob, cv2, os, time, PyPDF2
from pdf2image import convert_from_path


def user_set():
    disk = (12,         # Outer erosion of the mask. The higher it is, the more information is lost
                        # but the higher it fits to a bad image scan.
             8          # Inner erosion of the mask. lower than outer to loose less details in the
                        # center of the face.
             )
    lines_param = (40,  # Min line length. To increase with image size.
                   15   # Max line gap. To increase with image size.
                    )
    abs_dist = 2        # The tolerated gab of pixel in x axis between beginning and end of a vertical line. 
                        # The closer to 0, the more perfectly vertical your scanned lines have to be.
                        # With high sized images, it may be better to increase this value a bit
    
    delta_crop = 10     # The white pixels to add on your croped image size

    margins = 25        # The distance form the sides where no annotation should be taken into account
    
    duplicate_dist = 12 # The distance within which two contours are considered to be the same.
                        # You may want to increase it with the image size
                         
    circles_param = (20,# The min dist between circles. To increase with image size.
                     10,# The 2nd Hough estimator. The lower, the more circles it identifies.
                      2,# Min radius of a circle (depends on image size and the annotation wanted)
                     30 # Max radius of a circle
                    )

    shape_dist = 12     # The distance between a circle's center and a contour's center within which
                        # the contour is identified as this circle.
    return disk, lines_param, abs_dist, delta_crop, margins, duplicate_dist, circles_param, shape_dist    
    
    
def df_creator():
    """Creates the dataframe to store analysis results
    Returns the dataframe
    """
    coord_df = pd.DataFrame()
    coord_df[['file_id', 'file_name', 'tuple_coord', 'coord_x', 'coord_y', 'type']] = 0
    return coord_df


def delete_images(init=0):
    if init == 1:
        inputs_path = os.getcwd() + "/v2/"
    else:
        inputs_path = os.getcwd()
    os.chdir(inputs_path)
    png_files = glob.glob("*.png")
    for file in png_files:
        os.remove(os.path.join(inputs_path, file))
    os.chdir(os.pardir)
        

def pdf_collector():
   inputs_path = os.getcwd() + "/v2/"
   os.chdir(inputs_path)
   file = glob.glob("*.pdf")
   file.sort(key=os.path.getmtime)
   pdf_files = list(file)
   os.chdir(os.pardir)
   return pdf_files


def mask_dilution(im_cal, disk):
    """Modifies the reference image to create a mask which will be used on images to analyse
    Returns the mask
    """
    # make a copy of ref image
    dilated_mask = im_cal.copy()
    lim_left, lim_right = round(im_cal.shape[1]*1/4), round(im_cal.shape[1]*3/4)
    dil_left = morphology.erosion(im_cal[:, :lim_left], morphology.disk(disk[0]))
    dil_right = morphology.erosion(im_cal[:, lim_right:], morphology.disk(disk[0]))
    dil_inner = morphology.erosion(im_cal[:, lim_left:lim_right], morphology.disk(disk[1]))
    # update the mask with the eroded areas
    dilated_mask[:, lim_right:] = dil_right
    dilated_mask[:, :lim_left] = dil_left
    dilated_mask[:, lim_left:lim_right] = dil_inner
    # convert to numpy array
    dilated_mask = np.array(dilated_mask)*1
    return dilated_mask


def initialize(lines_param, abs_dist, disk, delta_crop):
    """Initializes the variables needed for the program
    Returns those variables
    """
    delete_images(init=1)
    im_cal = calibration(lines_param, abs_dist, delta_crop)
    coord_df = df_creator()
    file_id, images_to_test = 0, list()
    pdf_files = pdf_collector()
    dilated_mask = mask_dilution(im_cal, disk)
    return im_cal, coord_df, file_id, images_to_test, pdf_files, dilated_mask


def check_pdf_size(pdf):
    oversized = False
    inputs_path = os.getcwd() + "/v2/"
    os.chdir(inputs_path)
    writer = PyPDF2.PdfFileWriter()  # create a writer to save the updated results
    file = PyPDF2.PdfFileReader(pdf)
    for i in range(file.getNumPages()):
        p = file.getPage(i)
        h = p.mediaBox.getHeight()
        if h > 2000:                # issue if only one page is high size (but unlikely)
            oversized = True
            p.scaleBy(0.3)
            writer.addPage(p)
            with open("Resized_file.pdf", "wb+") as f:
                writer.write(f)
            pdf = writer
    return pdf, oversized


def pdf_to_png(pdf):
    pdf, oversized = check_pdf_size(pdf)
    img_list = list()
    if oversized == False:
      pdf_path = os.path.abspath(pdf)
    else:
      pdf_path = os.path.abspath("Resized_file.pdf")
    images = convert_from_path(pdf_path)  
    for count, img in enumerate(images):
      img_name = f"{pdf}_page_{count}.png"  
      img.save(img_name, "PNG", optimize=True)
      img_list.append(img_name)
    return img_list


def find_face(img_list):
    for i in range(len(img_list)):
        # Read the image
        img = cv2.imread(img_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect circular shape in the image
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  
        # using a findContours() function
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # selecting the "big" contour which is the face oval shape
        contours = [c for c in contours if len(c)>300]
        
        # list for storing names of shapes
        for contour in contours:
            
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            # more than 6 sides should be a circular shape
            if len(approx) > 6:
                np_im = np.array(img)
                img = Image.open(img_list[i])
                return img, np_im

    img = Image.open(img_list[i])
    np_im = np.array(img)
    return img, np_im


def bordure_detection(face, lines_param, abs_dist, cal):
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
              minLineLength=lines_param[0], # Min allowed length of line
              maxLineGap=lines_param[1] # Max allowed gap between line for joining them
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
          if y1 != y2 and np.abs(x1-x2) <= abs_dist:
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


def im_croping(X, Y, img, delta_crop):
    """Crops the image based on the min and max coordinates of the vertical lines which are the
     cadran of the face sketch, and then adds margins.
    Returns the newly cropped image
    """
    left = np.min(X)
    top = np.min(Y)
    right = np.max(X)
    bottom = np.max(Y)
    img_res = img.crop((left - delta_crop, top - delta_crop, right + delta_crop, bottom + delta_crop)) 
    return img_res


def calibration(lines_param, abs_dist, delta_crop):
    """Uses the reference image as a caliber for analyzed images 
    Returns the caliber image
    """
    path = os.getcwd()
    ref_path = path + "/Reference/Reference.PNG"
    sample = Image.open(ref_path)
    np_sample = np.array(sample)
    X_cal, Y_cal, error = bordure_detection(np_sample, lines_param, abs_dist, cal=1)
    if error == 1:
        print('There is an issue with the reference image.')
    im_with_face = im_croping(X_cal, Y_cal, sample, delta_crop)
    im_test_rgb = im_with_face.convert('RGB')

    # Convert into grayscale image
    im_test_gray = color.rgb2gray(im_test_rgb)
    thresh = threshold_otsu(im_test_gray)
    im_cal = im_test_gray > thresh
    return im_cal


def check_annotation(contour, cross_dots, margins):
    """Selects the mean coordinates of an annotation. Checks if they are out of margins area.
    Returns the check validation and the center coordinates
    """
    check = False
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


def duplicates_finder(contours_storage, duplicate_dist):
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
            elif center[0] >= (contours_storage[i][0] - duplicate_dist) and center[0] <= (contours_storage[i][0] + duplicate_dist):
                if center[1] >= (contours_storage[i][1] - duplicate_dist) and center[1] <= (contours_storage[i][1] + duplicate_dist):
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


def store_contours(cross_dots, margins, duplicate_dist):
    """Identifies all the contours in the image
    Returns the list of contours
    """
    cs = list()
    contours = measure.find_contours(cross_dots)
    for contour in contours:
        check, row_center, col_center = check_annotation(contour, cross_dots, margins)
        if check == True:
            cs.append((row_center, col_center))
    contours_storage = duplicates_finder(cs, duplicate_dist)
    return contours_storage


def circles_detector(cross_dots, circles_param, duplicate_dist):
    """Detects and stores the circle shape annotations in the image.
    Returns circle(s) center(s) list
    """
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 20
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.15
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(cross_dots)
    print(len(keypoints))
    circle_centers=list()
    if len(keypoints) > 0:
        for point in keypoints:
            a, b = np.uint16(np.around(point.pt[0])) , np.uint16(np.around(point.pt[1]))
            circle_centers.append((b,a))
    # find duplicates
    circle_centers = duplicates_finder(circle_centers, duplicate_dist)
    return circle_centers


def sep_cross(contours_storage, circle_centers, im_thresh, shape_dist):
    """Compares all the annotation and circles lists to keep only non-circles centers from the
     1st list. Allocates those centers to the cross(es) center(s) list.
    Returns the cross(es) center(s) list
    """
    cross_centers = contours_storage.copy()

    for i in range(len(cross_centers)):
        for j in range(len(circle_centers)):
            if (cross_centers[i][0] >= circle_centers[j][0]-shape_dist)  \
            and (cross_centers[i][1] >= circle_centers[j][1]-shape_dist) \
            and (cross_centers[i][0] <= circle_centers[j][0]+shape_dist) \
            and (cross_centers[i][1] <= circle_centers[j][1]+shape_dist):
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

def update_df(coord_df, circle_centers, cross_centers, file_id, local_im, images_to_test, im_thresh):
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
    # ident and clean variables
    delete_images()
    images_to_test.append(im_thresh)
    file_id+=1
    return coord_df


def non_nominal_update_df(coord_df, file_id, local_im, message):
    """Updates the results dataframe with information on the non-nominal result for this image
    Returns the updated dataframe
    """
    # update the global dataframe with non nominal case report
    df = pd.DataFrame(columns=['tuple_coord', 'coord_x', 'coord_y', 'type', 'file_id', 'file_name'])
    df.loc[0] = [np.nan, np.nan, np.nan, message, file_id, local_im]
    coord_df = pd.concat([coord_df, df])
    delete_images()
    return coord_df


def non_nominal_update(coord_df, file_id, pdf, images_to_test, im_thresh, message):
    print(message)
    coord_df = non_nominal_update_df(coord_df, file_id, pdf, message)
    images_to_test.append(im_thresh)
    file_id+=1
    return coord_df, images_to_test, file_id


def pd_to_csv(coord_df):
    """Transforms the pandas dataframe to a csv file and saves it.
    Returns nothing
    """
    path = os.getcwd()
    coord_df.to_csv(path+'/cross_circles_coord.csv', index=False)


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
        plt.savefig(path+"/Analysis_report/{}.jpeg".format(id))
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
    plt.savefig(path+"/Compliation.jpeg")
    plt.clf()


def main():
    # Initialize
    disk, lines_param, abs_dist, delta_crop, margins, duplicate_dist, circles_param, shape_dist = user_set()
    im_cal, coord_df, file_id, images_to_test, pdf_files, dilated_mask = initialize(lines_param, abs_dist, disk, delta_crop)
    # iterate through the pdf list
    for pdf in pdf_files:
        print('-------------- ', 'image {} over {}'.format(file_id+1, len(pdf_files)), ' --------------')
        img_list = pdf_to_png(pdf)
        img, np_im = find_face(img_list)
        
        # crop the image
        X, Y, error = bordure_detection(np_im, lines_param, abs_dist, cal=0)
        if error == 1:
            coord_df, images_to_test, file_id = non_nominal_update(coord_df, file_id, pdf,
                                                    images_to_test, im_thresh,
                                                    message='Did not manage to detect a face sketch.')
            continue
        img_res = im_croping(X, Y, img, delta_crop)

        # prepare the image and get the annotations
        im_thresh = image_threshold(img_res, im_cal)
        cross_dots = cross_cricles(im_thresh, dilated_mask)
        
        # no need to find contours if there is no mark
        if np.mean(cross_dots)==255:
            coord_df, images_to_test, file_id = non_nominal_update(coord_df, file_id, pdf,
                                                    images_to_test, im_thresh,
                                                    message='No annotation.')
            continue

        contours_storage = store_contours(cross_dots, margins, duplicate_dist)
        # if there are too many marks we assume there is a problem reading the file
        if len(contours_storage) > 10:
            coord_df, images_to_test, file_id = non_nominal_update(coord_df, file_id, pdf,
                                                    images_to_test, im_thresh,
                                                    message='Too many annotations detected.')
            continue

        # find the circles
        circle_centers = circles_detector(cross_dots, circles_param, duplicate_dist)

        # determine the cross as non-circle annotations
        cross_centers = sep_cross(contours_storage, circle_centers, im_thresh, shape_dist)

        # get the coordinates in a pandas dataframe
        if len(circle_centers)==0 and len(cross_centers)==0:
            coord_df, images_to_test, file_id = non_nominal_update(coord_df, file_id, pdf,
                                                    images_to_test, im_thresh,
                                                    message='There is no annotation.')
        else:
            coord_df = update_df(coord_df, circle_centers, cross_centers, file_id, pdf,
                                 images_to_test, im_thresh)
        
    # save the whole dataframe as an excel file
    pd_to_csv(coord_df)

    # verify the results
    print('Plotting results in progress...')
    verification_by_image(images_to_test, coord_df)
    compil_cross_and_circles(coord_df, im_cal)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    m = (end-start)//60
    s = np.round(end-start - m*60, 0) 
    print("Execution time: {} min {} sec".format(m, s))
