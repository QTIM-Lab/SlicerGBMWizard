from __future__ import division

import numpy as np
import nibabel as nib
import os
from shutil import copy, move
import matplotlib.pyplot as plt
import glob
from scipy import stats, signal, misc
from scipy.ndimage.morphology import binary_fill_holes
import csv
import fnmatch

def return_nifti_attributes(filepath):

    """ This returns nibabel's version of the nifti header. Note that this
        is NOT the full header! fslhd, from FSL, returns other things such
        as the qform orientation code. TODO: add an option for returning fslhd
        if installed.
    """

    img_nifti = nib.load(filepath)
    return img_nifti.header

def get_nifti_affine(input_data):

    if isinstance(input_data, basestring):
        affine = nib.load(input_data).affine
    else:
        affine = input_data.affine

    return affine

def set_nifti_affine(input_data, new_affine, output_filepath=None):

    """ This function takes in either a nibabel nifti object or a nifti
        filepath and alters its affine matrix.

        Parameters
        ----------
        input_data: str or nibabel object
            If str, will be interpreted as a filepath. Otherwise, will be
            interpreted as a nibabel nifti object.
        new_affine: 4x4 numpy array
            The new affine matrix to be set.
        output_filepath: str
            The output_filepath, if not the same as the input_filepath.

    """

    if isinstance(input_data, basestring):
        if output_filepath is None:
            output_filepath = input_data
        input_data = nib.load(input_data)
        output_nifti = nib.Nifti1Image(input_data.get_data(), new_affine)
        nib.save(output_nifti, output_filepath)
    else:
        input_data.affine = new_affine

def nifti_resave(input_filepath, output_filepath):

    """ Copies a file somewhere else. Effectively only used for compressing nifti files.

        Parameters
        ----------
        input_filepath: str
            Input filepath.
        output_filepath: str
            Output filepath to be copied to.

    """

    nib.save(nib.load(input_filepath), output_filepath)

def nifti_2_numpy(input_filepath, return_header=False):

    """ Copies a file somewhere else. Effectively only used for compressing nifti files.

        Parameters
        ----------
        input_filepath: str
            Input filepath.
        return_header: bool
            If true, returns header information in nibabel format.

        Returns
        -------
        img: Numpy array
            Untransformed image data.
        header: list
            A two item list. The first is the affine matrix in array format, the
            second is 

    """


    nifti = nib.load(input_filepath)

    if return_header:
        return nifti.get_data(), [nifti.affine, nifti.header]
    else:
        return nifti.get_data()

def create_4d_nifti_from_3d(input_4d_numpy, reference_nifti_filepath, output_filepath):

    """ Sometimes, a reference nifti is only available in 3D form when trying to
        generate a 4D volume. This function addresses that.
    """

    nifti_image = nib.load(reference_nifti_filepath)
    image_affine = nifti_image.affine
    nifti_image.header['dim'][0] = 4
    nifti_image.header['dim'][4] = input_4d_numpy.shape[-1]
    output_nifti = nib.Nifti1Image(input_4d_numpy, image_affine)
    nib.save(output_nifti, output_path)

def save_3d_numpy_from_4d_nifti(image_numpy, reference_nifti, output_filepath):

    return

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath=None, output_filepath=None):

    """ This is a bit convoluted.

        TODO: Documentation, rearrange reference_nifti and output_filepath, and
        propagate changes to the rest of qtim_tools.
    """

    if reference_nifti_filepath is not None:
        if isinstance(reference_nifti_filepath, basestring):
            nifti_image = nib.load(reference_nifti_filepath)
            image_affine = nifti_image.affine
        else:
            image_affine = reference_nifti_filepath
    else:
        print 'Warning: no reference nifti file provided. Generating empty header.'
        image_affine = np.eye(4)

    output_nifti = nib.Nifti1Image(image_numpy, image_affine)

    if output_filepath is None:
        return output_nifti
    else:
        nib.save(output_nifti, output_filepath)

""" All functions below will eventually be moved into other modules. They remain
    for now, because it takes time to figure out which other functions reference
    them.
"""

def coerce_levels(image_numpy, levels=255, method="divide", reference_image = [], reference_norm_range = [.075, 1], mask_value=0, coerce_positive=True):

    """ In volumes with huge outliers, the divide method will
        likely result in many zero values. This happens in practice
        quite often. TO-DO: find a better method to bin image values.
        I'm sure there are a thousand such algorithms out there to do
        so. Maybe something based on median's, rather than means. This,
        of course, loses the 'Extremeness' of extreme values. An open
        question of how to reconcile this -- maybe best left to the user.
        Note that there is some dubious +1s and -1s in this function. It
        may be better to clean these up in the future. I have also built-in
        the coerce-positive function into this function. The other one
        was not working for mysterious reasons.
    """

    if np.min(image_numpy) < 0 and coerce_positive:
        reference_image -= np.min(image_numpy)
        image_numpy[image_numpy != mask_value] -= np.min(image_numpy)

    levels -= 1
    if method == "divide":
        if reference_image == []:
            image_max = np.max(image_numpy)
        else:
            image_max = np.max(reference_image)
        for x in xrange(image_numpy.shape[0]):
            for y in xrange(image_numpy.shape[1]):
                for z in xrange(image_numpy.shape[2]):
                    if image_numpy[x,y,z] != mask_value:
                        image_numpy[x,y,z] = np.round((image_numpy[x,y,z] / image_max) * levels) + 1

    """ Another method is to bin values based on their z-score. I provide
        two options: within-ROI normalization, and whole-image normalization.
        The output is always the ROI image, but in the latter option z-scores
        are generated from the range of intensities across the entire image
        within some range of percentages. This range is currently determined
        from the mean, but it may make more sense to do it from the median;
        this protects the algorithm from extreme values. On the other hand,
        using the median could white out an otherwise heterogenous hotspot.
    """

    if method == "z_score":

        # check_image_2d(image_numpy, mode="maximal_slice", mask_value=mask_value)

        ## Note that this is a bad way to check this variable.
        if reference_image == []:
            masked_image_numpy = np.ma.masked_equal(image_numpy, mask_value)
            z_image_numpy = stats.zscore(masked_image_numpy, axis=None)

            # image_range = [np.min(z_image_numpy), np.max(z_image_numpy)]
            image_range = [np.mean(z_image_numpy) - np.std(z_image_numpy), np.mean(z_image_numpy) + np.std(z_image_numpy)]
            bins = np.linspace(image_range[0], image_range[1], levels)

            # distribution = stats.norm(loc=np.mean(z_image_numpy), scale=np.var(z_image_numpy))

            # # percentile point, the range for the inverse cumulative distribution function:
            # bounds_for_range = distribution.cdf([0, 100])

            # # Linspace for the inverse cdf:
            # pp = np.linspace(*bounds_for_range, num=levels)

            # bins = distribution.ppf(pp)
            # print bins
        else:
            masked_reference_image = np.ma.masked_equal(reference_image, mask_value)
            masked_reference_image = np.ma.masked_less(masked_reference_image, reference_norm_range[0]*np.max(reference_image))
            masked_reference_image = np.ma.masked_greater(masked_reference_image, reference_norm_range[1]*np.max(reference_image))
            masked_image_numpy = np.ma.masked_equal(image_numpy, mask_value)
            z_image_numpy = stats.zmap(masked_image_numpy, masked_reference_image, axis=None)

            z_reference_image = stats.zscore(masked_reference_image, axis=None)

            # distribution = stats.norm(loc=np.mean(z_reference_image), scale=np.var(z_reference_image))

            # # percentile point, the range for the inverse cumulative distribution function:
            # bounds_for_range = distribution.cdf([0, 100])

            # # Linspace for the inverse cdf:
            # pp = np.linspace(*bounds_for_range, num=levels)

            # bins = distribution.ppf(pp)

            # image_range = [np.mean(z_reference_image) - np.std(z_reference_image), np.mean(z_reference_image) + np.std(z_reference_image)]
            image_range = [np.min(z_reference_image), np.max(z_reference_image)]
            bins = np.linspace(image_range[0], image_range[1], levels)

        for x in xrange(image_numpy.shape[0]):
            for y in xrange(image_numpy.shape[1]):
                for z in xrange(image_numpy.shape[2]):
                    if image_numpy[x,y,z] != mask_value:
                        image_numpy[x,y,z] = (np.abs(bins-z_image_numpy[x,y,z])).argmin() + 1

        # check_image_2d(image_numpy, mode="maximal_slice", mask_value=mask_value)
    image_numpy[image_numpy == mask_value] = 0
    return image_numpy

def coerce_positive(image_numpy):

    """ Required by GLCM. Not sure of the implications for other algorithms.
    """

    image_min = np.min(image_numpy)
    if image_min < 0:
        image_numpy = image_numpy - image_min
    return image_numpy

def remove_islands():
    return

def erode_label(image_numpy, iterations=2, mask_value=0):

    """ For each iteration, removes all voxels not completely surrounded by
        other voxels. This might be a bit of an aggressive erosion. Also I
        would bet it is incredibly ineffecient. Also custom erosions in
        multiple dimensions look a little bit messy.
    """

    iterations = np.copy(iterations)

    if isinstance(iterations, list):
        if len(iterations) != 3:
            print 'The erosion parameter does not have enough dimensions (3). Using the first value in the eroison parameter.'
    else:
        iterations == [iterations, iterations, iterations]

    for i in xrange(max(iterations)):

        kernel_center = 0
        edges_kernel = np.zeros((3,3,3),dtype=float)
        
        if iterations[2] > 0:
            edges_kernel[1,1,0] = -1
            edges_kernel[1,1,2] = -1
            iterations[2] -= 1
            kernel_center += 2

        if iterations[1] > 0:
            edges_kernel[1,0,1] = -1
            edges_kernel[1,2,1] = -1
            iterations[1] -= 1
            kernel_center += 2

        if iterations[0] > 0:
            edges_kernel[0,1,1] = -1
            edges_kernel[2,1,1] = -1
            iterations[0] -= 1
            kernel_center += 2

        edges_kernel[1,1,1] = kernel_center

        label_numpy = np.copy(image_numpy)
        label_numpy[label_numpy != mask_value] = 1
        label_numpy[label_numpy == mask_value] = 0

        edge_image = signal.convolve(label_numpy, edges_kernel, mode='same')
        edge_image[edge_image < 0] = -1
        edge_image[np.where((edge_image <= kernel_center) & (edge_image > 0))] = -1
        edge_image[edge_image == 0] = 1
        edge_image[edge_image == -1] = 0
        image_numpy[edge_image == 0] = mask_value

    return image_numpy

def check_image_2d(image_numpy, second_image_numpy=[], slice_view="cycle", step=1, mask_value=0, slice_axis="first"):

    """ A useful utiltiy for spot checks. TODO: Add utility for dynamic axis viewing. Also TODO: make
        a check_image_3d
    """

    if second_image_numpy != []:
        for i in xrange(image_numpy.shape[0]):
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(second_image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            plt.show()
    else:
        if slice_view == "cycle":
            for i in xrange(image_numpy.shape[0]):
                fig = plt.figure()
                imgplot = plt.imshow(image_numpy[i,:,:], interpolation='none', aspect='auto')
                plt.show()

        elif slice_view == "first":
            fig = plt.figure()
            imgplot = plt.imshow(image_numpy[0,:,:], interpolation='none', aspect='auto')
            plt.show()

        elif slice_view == "maximal_slice":

            maximal = [0, np.zeros(image_numpy.shape)]

            for i in xrange(image_numpy.shape[2]):
            
                image_slice = image_numpy[:,:,i]

                test_maximal = (image_slice != mask_value).sum()

                if test_maximal >= maximal[0]:
                    maximal[0] = test_maximal
                    maximal[1] = image_slice

            fig = plt.figure()
            imgplot = plt.imshow(maximal[1], interpolation='none', aspect='auto')
            plt.show()

        else:
            fig = plt.figure()
            imgplot = plt.imshow(image_numpy[slice_view,:,:], interpolation='none', aspect='auto')
            plt.show()

def check_tumor_histogram(image_numpy, second_image_numpy=[], mask_value=0, image_name = ''):

    """ TODO: Make more general, edit out the word tumor
    """

    if second_image_numpy != []:
        tumor_ROI = image_numpy[image_numpy != mask_value]
        whole_brain = second_image_numpy[second_image_numpy > (second_image_numpy.max()*0.075)]
        bins = np.linspace(second_image_numpy.max()*0.075, second_image_numpy.max(), 100)

        fig = plt.figure()

        brain_hist = fig.add_subplot(2,1,1)
        brain_hist.hist(whole_brain, bins, alpha=1, label="Whole Brain > 7.5 Percent Intensity")
        if image_name != '':
            plt.title(image_name)

        tumor_hist = fig.add_subplot(2,1,2)
        tumor_hist.hist(tumor_ROI, bins, alpha=1, label='Tumor ROI(s)', color='red')
        plt.title('Tumor ROI(s)')

        # plt.legend(loc='upper right')
        # plt.show()
        if image_name[0:3] == 'TMZ':
            plt.savefig(image_name + '.png')
        else:
            plt.savefig("Melanoma_" + image_name + '.png')
        plt.close()

    else:
        tumor_ROI = image_numpy[image_numpy != mask_value]
        plt.hist(tumor_ROI)
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        fig = plt.gcf()

        plt.show()


def assert_3D(image_numpy):
    if len(image_numpy.shape) > 3:
        if len(image_numpy.shape) == 4 and image_numpy.shape[3] == 1:
            image_numpy = image_numpy[:,:,:,0]
        else:
            return False

    return True

def assert_nD(array, ndim, arg_name='image'):

    """

    This script is currently directly copied from scikit-image.

    Verify an array meets the desired ndims.
    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.
    """
    array = np.asanyarray(array)
    msg = "The parameter `%s` must be a %s-dimensional array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if not array.ndim in ndim:
        raise ValueError(msg % (arg_name, '-or-'.join([str(n) for n in ndim])))

def fill_in_convex_outline(filepath, output_file, outline_lower_threshold=[], outline_upper_threshold=[], outline_color=[], output_label_num=1, reference_nifti=[]):

    outline_upper_threshold = np.array(outline_upper_threshold)
    outline_lower_threshold = np.array(outline_lower_threshold)

    if filepath.endswith('.nii') or filepath.endswith('nii.gz'):
        return

    else:
        image_file = misc.imread(filepath)
        label_file = np.zeros_like(image_file)
        # print image_file.shape

        for row in xrange(image_file.shape[0]):
            row_section = 0
            fill_index = 0
            for col in xrange(image_file.shape[1]):
                match = False
                pixel = image_file[row, col, ...]
                
                if outline_upper_threshold != [] and outline_lower_threshold != []:
                    if all(pixel > outline_lower_threshold) and all(pixel < outline_upper_threshold):
                        match = True
                elif outline_color != []:
                    if (pixel == outline_color):
                        match = True
                else:
                    print 'Error. Please provide a valid outline color or threshold.'
                    return

                if match:
                    label_file[row, col, ...] = output_label_num                    

        if reference_nifti == []:
            label_file = binary_fill_holes(label_file[:,:,0]).astype(label_file.dtype)*255
            misc.imsave(output_file, label_file)
        else:
            label_file = label_file[:,:,0]
            save_numpy_2_nifti(label_file, reference_nifti, output_file)

def replace_slice(input_nifti_slice_filepath, reference_nifti_filepath, output_file, slice_number, orientation_commands=[np.rot90, np.flipud]):

    """ Orients a 1-D label nifti, likely created by the convex outline function, with respect to a reference
        3-D nifti. Saves out a 3-D nifti of the same shape with just the one ROI slice.
    """

    refererence_numpy = nifti_2_numpy(reference_nifti_filepath)
    input_numpy_slice = nifti_2_numpy(input_nifti_slice_filepath)
    output_numpy = np.zeros_like(refererence_numpy)

    # Some rotations may be necessary to get the labelmap in the right orientation.
    # It is not obvious from the starting jpgs what those rotations should be.

    for transformation_function in orientation_commands:
        input_numpy_slice = transformation_function(input_numpy_slice)

    output_numpy[:,:,slice_number] = input_numpy_slice[:,:]

    save_numpy_2_nifti(output_numpy, reference_nifti_filepath, output_file)

def truncate_image(input_volume, mask_value=0, return_mask=False, padding=0, output_mask_filename=""):

    """ This function takes in an N-dimensional array and truncates all rows/columns/etc
        that contain only mask values. Useful for reducing computation time on functions
        whose running time scales exponentially with dimensions size.

        BUG: Currently seems to fail on axes with length 1.
        TODO: Truncate only on some axes.
        TODO: Add the option to add buffer pixels to meet output_dimensions.

        Parameters
        ----------

        input_volume: N-dimensional array
            The volume to be truncated. Will be truncated in every axis.
        mask_value: int or float
            Vectors in an axis that are composed entirely of mask_value will be truncated.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    dims = image_numpy.shape
    truncate_ranges = [[0, 0] for x in dims]

    for axis, axis_length in enumerate(dims):
        start_flag = True
        for idx in range(axis_length):
            if (get_arbitrary_axis_slice(image_numpy, axis, idx) == mask_value).all():
                if start_flag:
                    truncate_ranges[axis][0] = idx + 1
            else:
                start_flag = False
                truncate_ranges[axis][1] = idx + 1

    if padding > 0:
        truncate_ranges = [[max(0, x[0]-padding), min(dims[axis], x[1]+padding)] for axis, x in enumerate(truncate_ranges)]

    truncate_slices = [slice(x[0], x[1]) for x in truncate_ranges]

    truncate_image_numpy = image_numpy[truncate_slices]

    if return_mask:
        return truncate_image_numpy, mask_numpy
    else:
        return truncate_image_numpy

def split_image(input_volume, input_label_volume=None, label_indices=None, mask_value=0):

    """ This function takes in an image, optionally a label image, and optionally a set of indices,
        and returns one duplicate masked image for each given label. Useful for analyzing,
        say, multiple tumors, although expensive in memory. Useful when paired with the
        truncate_image function to reduce array memory.
    """

    image_numpy = convert_input_2_numpy(input_volume)
    label_numpy = convert_input_2_numpy(input_label_volume)

    masked_images = []

    print label_indices

    if label_indices is None:
        if label_numpy is None:
            label_indices = np.unique(image_numpy)
        else:
            label_indices = np.unique(label_numpy)

    if mask_value in label_indices:
        label_indices = np.delete(np.array(label_indices), np.argwhere(label_indices==mask_value))

    for idx in label_indices:
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = mask_value
        masked_images += [masked_image]

    return masked_images

def extract_maximal_slice(input_volume, input_label_volume='', mode='max_intensity', axis=2, mask_value=0, return_index=False):

    """ Extracts one slice from a presumably 3D volume. Either take the slice whose label
        has the greatest area (mode='max_label'), or whos sum of voxels has the greatest 
        intensity (mode='max_intensity'), according to the provided axis variable.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    sum_dimensions = tuple([int(x) for x in range(0,image_numpy.ndim) if x != axis])

    if mode == 'max_intensity':
        flattened_image = np.sum(image_numpy, axis=sum_dimensions)
    elif mode == 'max_label':
        label_numpy = convert_input_2_numpy(input_label_volume)
        flattened_image = np.sum(label_numpy, axis=sum_dimensions)
    elif mode == 'non_mask':
        flattened_image = (image_numpy != mask_value).sum(axis=sum_dimensions)
    else:
        print 'Invalid mode entered to extract_maximal_slice_3d. Returning original array..'
        return image_numpy

    # TODO: Put in support for 
    highest_slice_index = np.argmax(flattened_image)
    try:
        highest_slice_index = highest_slice_index[0]
    except:
        pass

    if return_index:
        return get_arbitrary_axis_slice(image_numpy, axis, highest_slice_index), highest_slice_index

    return get_arbitrary_axis_slice(image_numpy, axis, highest_slice_index)

def grab_files_recursive(input_directory, regex='*'):

    """ Returns all files recursively in a directory. Essentially a convenience wrapper 
        around os.walk.

        Parameters
        ----------

        input_directory: str
            The folder to search.
        regex: str
            A linux-style pattern to match.

        Returns
        -------
        output_list: list
            A list of found files.
    """

    output_list = []

    for root, subFolders, files in os.walk(input_directory):
        for file in files:
            if fnmatch.fnmatch(file, regex):
                output_list += [os.path.join(root, file)]

    return output_list


# Consider merging these into one dictionary. Separating them
# is easier to visaulize though.
FORMAT_LIST = {'nifti':('.nii','.nii.gz')}

NUMPY_CONVERTER_LIST = {'nifti':nifti_2_numpy}

def check_format(filepath):

    format_type = None

    for data_type in FORMAT_LIST:
        if filepath.lower().endswith(FORMAT_LIST[data_type]):
            format_type = data_type
        if format_type is not None:
            break

    if format_type is None:
        print 'Error! Input file extension is not supported by qtim_tools. Returning None.'
    else:
        return format_type

def convert_input_2_numpy(input_data, input_format=None, return_header=False, return_type=False):
    
    """ Copies a file somewhere else. Effectively only used for compressing nifti files.

        Parameters
        ----------
        input_filepath: str
            Input filepath.
        return_header: bool
            If true, returns header information in nibabel format.

        Returns
        -------
        img: Numpy array
            Untransformed image data.
        header: list
            Varies from format to format.
        type: str
            Internal code for image type.
    """

    return_items = []

    if isinstance(input_data, basestring):
        if input_format is None:
            input_format = check_format(input_data)

        if input_format is None:
            print 'Cannot understand input format for numpy conversion, returning None.'
            if return_header:
                return None, None
            else:
                return None

        if return_header:
            return_items += NUMPY_CONVERTER_LIST[input_format](input_data, return_header=True)
        else:
            return_items = [NUMPY_CONVERTER_LIST[input_format](input_data)]
        if return_type:
            return_items += [input_format]

    else:
        return_items += [input_data]
        if return_header:
            return_items += [None]
        if return_type:
            return_items += ['numpy']

    if len(return_items) > 1:
        return return_items
    else:
        return return_items[0]

def crop_with_mask(input_data, label_data, output_filename='', mask_value=0, return_labels=None, replacement_value=0):

    """ Crops and image with a predefined mask image. Values equal to mask value
        are replaced with replacement_value.

        TODO: Add support for not-equal-to masking.
        TODO: Add support for other replacement_values, a-la scikit-learn.
        TODO: Add support for multiple return_labels.

        Parameters
        ----------

        input_data: N-dimensional array or str
            The volume to be cropped. Can be filename or numpy array.
        label_data: N-dimensional array or str
            A label mask. Must be the same size as input_data
        mask_value: int or float
            Values equal to mask_value will be replaced with replacement_value.
        return_labels: float or list
            Label values to be preserved at the exclusion of other values.
        replacement_value: int or float
            Values equal to mask_value or not in return_labels will be replaced with replacement_value.

        Returns
        -------
        input_numpy: array
            Transformed input array.
    """

    input_numpy, label_numpy = convert_input_2_numpy(input_data), convert_input_2_numpy(label_data)

    input_numpy[label_numpy == mask_value] = replacement_value

    # if not isinstance(return_labels, list): 
    #     return_labels = [return_labels]

    # if len(return_labels) > 0:
    #     # [TODO FIX]
    #     input_numpy[label_numpy != return_labels[0]] = replacement_value

    if output_filename != '':
        if isinstance(input_data, basestring):
            save_numpy_2_nifti(input_numpy, input_data, output_filename)
        else:
            save_numpy_2_nifti(input_numpy, output_path=output_filename)
    else:
        return input_numpy

if __name__ == '__main__':

    pass