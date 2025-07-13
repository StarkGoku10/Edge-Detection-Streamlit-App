#!/usr/bin/env python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import skimage.transform
import scipy.stats as st
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

# Helper function for deterministic colormap
def get_deterministic_colormap(num_colors):
    """
    Generates a list of unique, deterministic colors.
    Parameters:
        num_colors (int): The number of unique colors needed.
    Returns:
        np.ndarray: An array of RGB colors.
    """
    # Using a perceptually uniform colormap like viridis and spacing out the colors
    colormap = plt.cm.get_cmap('viridis', num_colors)
    # Get RGB, discard alpha
    colors = [colormap(i)[:3] for i in range(num_colors)] 

    return np.array(colors)

def Gaussiankernel2D(kernelsize, sigma):
    """
    Generate a 2D Gaussian kernel.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: 2D Gaussian kernel.
    """
    spacing = (2*sigma+1)/kernelsize
    x = np.linspace((-sigma-spacing)/2, (sigma+spacing)/2, kernelsize+1)
    kernel1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kernel1d, kernel1d))
    kernel = kernel_raw/kernel_raw.sum()

    return kernel

def Gaussian1D(sigma, mean, x, order_derivative):
    """
    Generate a 1D Gaussian function or its derivatives.
    Parameters:
        sigma (float): Standard deviation.
        mean (float): Mean of the Gaussian.
        x (np.ndarray): Range of values.
        order_derivative (int): Derivative order (0, 1, or 2).
    Returns:
        np.ndarray: 1D Gaussian or its derivative.
    """
    x = np.array(x)
    x_ = x - mean
    vari = sigma**2
    g1 = np.exp(-0.5 * (x_**2 / vari)) / np.sqrt(2*np.pi*vari)
    if order_derivative == 0:
        return g1
    elif order_derivative == 1:
        return -g1 * (x_ / vari)
    else:
        return g1 * ((x_**2 - vari) / (vari**2))

def Gaussian2D(kernelsize, sigma):
    """
    Generate a 2D Gaussian function.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: 2D Gaussian kernel
    """
    vari = sigma * sigma
    shape = (kernelsize,kernelsize)
    n,m = [(i - 1)//2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = np.exp(-(x*x + y*y) / (2*vari)) / (2*np.pi*vari)

    return g / g.sum()

def lap_gaussian2D(kernelsize, sigma):
    """
    Generate a 2D Laplacian of Gaussian (LoG) kernel.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: Laplacian of Gaussian kernel.
    """
    vari = sigma * sigma
    shape = (kernelsize,kernelsize)
    n,m = [(i - 1)//2 for i in shape]
    x, y = np.ogrid[-m:m+1, -n:n+1]
    g = np.exp(-(x*x + y*y) / (2*vari)) / (2*np.pi*vari)
    h = g * ((x*x + y*y) - vari) / (vari**2)
    h = h / (np.sum(np.abs(h)) + 1e-10)

    return h

def makefilter(sigma, x_orient, y_orient, pts, kernelsize):
    """
    Create a 2D filter using Gaussian derivatives.
    Parameters:
        sigma (float): Scale of the Gaussian.
        x_orient (int): Orientation for x-derivative.
        y_orient (int): Orientation for y-derivative.
        pts (np.ndarray): Grid of points.
        kernelsize (int): Size of the kernel.
    Returns:
        np.ndarray: Filter kernel.
    """
    gx = Gaussian1D(3*sigma, 0, pts[0,...], x_orient)
    gy = Gaussian1D(sigma,   0, pts[1,...], y_orient)
    image = gx*gy
    image = np.reshape(image,(kernelsize,kernelsize))
    im_max = np.max(np.abs(image)) + 1e-10 # Adding epsilon
    image = image / im_max

    return image

def Oriented_DOG(sigma, orient, size):
    """
    Generate an oriented Difference of Gaussian (DoG) filter bank.
    Parameters:
        sigma (list of float): List of sigma values for scales.
        orient (int): Number of orientations.
        size (int): Size of the kernel.
    Returns:
        list: List of DoG filters.
    """
    kernels = []
    border = cv2.BORDER_DEFAULT
    for scale_val in sigma: 
        orients = np.linspace(0,360,orient, endpoint=False)
        kernel = Gaussiankernel2D(size,scale_val)
        sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=3, borderType=border)
        smax = np.max(np.abs(sobelx64f)) + 1e-10 # Adding epsilon
        sobelx64f /= smax
        for eachOrient in orients:
            rotated = skimage.transform.rotate(sobelx64f, eachOrient, mode='reflect')
            rmax = np.max(np.abs(rotated)) + 1e-10 # Adding epsilon
            rotated /= rmax
            kernels.append(rotated)

    return kernels

def LM_filters(kernelsize, sigma_scales, num_orientations, nrotinv): 
    """
    Generate Leung-Malik (LM) filter bank.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma_scales (int): Number of scales.
        num_orientations (int): Number of orientations.
        nrotinv (int): Number of rotationally invariant filters.
    Returns:
        np.ndarray: Array of LM filters.
    """
    scalex = np.sqrt(2) * np.arange(1,sigma_scales+1)
    nbar = len(scalex)*num_orientations
    nedge = len(scalex)*num_orientations
    nf = nbar+nedge+nrotinv
    F = np.zeros([kernelsize,kernelsize,nf], dtype=np.float32)
    hkernelsize = (kernelsize - 1)/2

    x_range = np.arange(-hkernelsize, hkernelsize+1)
    x,y = np.meshgrid(x_range,x_range)
    orgpts = np.vstack([x.flatten(), y.flatten()])
    count = 0

    for scaleVal in scalex:
        for orient_idx in range(num_orientations): 
            angle = (math.pi * orient_idx)/num_orientations
            c = np.cos(angle)
            s = np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
            rotpts = rot_mat.dot(orgpts)
            F[:,:,count] = makefilter(scaleVal, 0, 1, rotpts, kernelsize)
            F[:,:,count+nedge] = makefilter(scaleVal, 0, 2, rotpts, kernelsize)
            count = count + 1
    count = nbar+nedge 
    sigma_vals_invariant = np.sqrt(2) * np.array([1,2,3,4], dtype=np.float32) 

    for i in range(len(sigma_vals_invariant)):
        g2d = Gaussian2D(kernelsize, sigma_vals_invariant[i])
        g2d /= (np.max(np.abs(g2d)) + 1e-10) # Adding epsilon
        F[:, :, count] = g2d
        count += 1

    for i in range(len(sigma_vals_invariant)):
        log_2d = lap_gaussian2D(kernelsize, sigma_vals_invariant[i])
        F[:, :, count] = log_2d
        count += 1

    for i in range(len(sigma_vals_invariant)): 
        log_2d = lap_gaussian2D(kernelsize, 3*sigma_vals_invariant[i])
        F[:, :, count] = log_2d
        count += 1

    return F

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    """
    Generates a Gabor filter kernel.
    Parameters:
        sigma (float): Standard deviation of the Gaussian envelope.
        theta (float): Orientation of the Gabor filter (in radians).
        Lambda (float): Wavelength of the sinusoidal carrier.
        psi (float): Phase offset of the sinusoidal carrier.
        gamma (float): Spatial aspect ratio (ellipticity) of the Gabor filter.
    Returns:
        np.ndarray: A 2D Gabor filter kernel.
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    nstds = 3
    x_max_abs = max(abs(nstds*sigma_x*np.cos(theta)),
                    abs(nstds*sigma_y*np.sin(theta)))
    x_max = int(np.ceil(x_max_abs))

    y_max_abs = max(abs(nstds*sigma_x*np.sin(theta)),
                    abs(nstds*sigma_y*np.cos(theta)))
    y_max = int(np.ceil(y_max_abs))

    x_min, y_min = -x_max, -y_max
    (y, x) = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(
        -0.5 * (x_theta**2 / (sigma_x**2 + 1e-10) + y_theta**2 / (sigma_y**2 + 1e-10)) # Epsilon for stability
    ) * np.cos(2*np.pi / Lambda * x_theta + psi)
    denom = np.sqrt((gb**2).sum()) + 1e-10 # Adding epsilon
    gb = gb / denom

    return gb

def Gabor_filter(sigma_list, theta, Lambda, psi, gamma, num_filters):
    """
    Creates a set of Gabor filters with varying orientations.
    Parameters:
        sigma_list (list): List of standard deviations for the Gaussian envelope.
        theta (float): Fixed orientation for the base Gabor filter.
        Lambda (float): Wavelength of the sinusoidal carrier.
        psi (float): Phase offset of the sinusoidal carrier.
        gamma (float): Spatial aspect ratio (ellipticity) of the Gabor filter.
        num_filters (int): Number of filters to generate with different orientations.
    Returns:
        list: A list of 2D Gabor filter kernels.
    """
    g_filters = [] 
    ang = np.linspace(0, 360, num_filters, endpoint=False)
    for s_val in sigma_list: 
        base = gabor_fn(s_val, theta, Lambda, psi, gamma)
        for a_val in ang: 
            image = skimage.transform.rotate(base,a_val,mode='reflect')
            imax = np.max(np.abs(image)) + 1e-10 
            image /= imax
            g_filters.append(image)

    return g_filters

def textonmap_DOG(Img, filter_bank):
    """
    Generates a texton map using a Difference of Gaussian (DoG) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (list): List of DoG filter kernels.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    if len(Img.shape) == 3:
        Img_gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    else:
        Img_gray = Img

    # Initialize with the original image for dstack
    tex_map_list = [Img.copy()] 

    for i in range(len(filter_bank)):
        # Ensure filter is float64 for cv2.filter2D compatibility
        filt = filter_bank[i].astype(np.float64)
        out = cv2.filter2D(Img_gray,-1,filt)
        tex_map_list.append(out)

    return np.dstack(tex_map_list)


def textonmap_LM(Img, filter_bank):
    """
    Generates a texton map using an LM (Leung-Malik) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (np.ndarray): A 3D array representing the LM filter bank.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    if len(Img.shape) == 3:
        Img_gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    else:
        Img_gray = Img
    # Initialize with the original image for dstack
    tex_map_list = [Img_gray.copy()] 

    _,_,num_filters = filter_bank.shape
    for i in range(num_filters):
        # Ensure filter is float64 for cv2.filter2D compatibility
        filt = filter_bank[:,:,i].astype(np.float64)
        out = cv2.filter2D(Img_gray,-1,filt)
        tex_map_list.append(out)

    return np.dstack(tex_map_list)


def Texton(img, filter_bank1, filter_bank2, filter_bank3, num_clusters):
    """
    Computes a texton map by applying multiple filter banks and clustering responses.
    Parameters:
        img (np.ndarray): Input color image.
        filter_bank1 (np.ndarray): LM filter bank.
        filter_bank2 (list): DoG filter bank.
        filter_bank3 (list): Gabor filter bank.
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        tuple: (2D label map, color visualization)
    """
    # Ensure img is float for processing, typically [0,1] or [0,255]
    # Normalizing to [0,1] if it's uint8 [0,255]
    if img.dtype == np.uint8:
        img_proc = img.astype(np.float32) / 255.0
    # Assuming it might be float but not normalized
    elif img.max() > 1.0: 
        img_proc = img.astype(np.float32) / 255.0
    else:
        img_proc = img.astype(np.float32)

    p,q,_ = img_proc.shape
    weights = [0.3,0.3,0.4]

    tex_map_DOG = textonmap_DOG(img_proc, filter_bank2) * weights[0]
    tex_map_LM = textonmap_LM(img_proc, filter_bank1) * weights[1]
    tex_map_Gabor = textonmap_DOG(img_proc, filter_bank3) * weights[2]

    # The first slice of each texton map is the original image, skip it.
    combined = np.dstack((
        tex_map_DOG[:, :, 1:],
        tex_map_LM[:, :, 1:],
        tex_map_Gabor[:, :, 1:]
    ))

    m,n,r = combined.shape
    inp = np.reshape(combined,((p*q),r))
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000, n_init='auto')
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    label_map = np.reshape(labels,(m,n))

    # Create deterministic color visualization
    colors = get_deterministic_colormap(num_clusters)
    # Directly map labels to colors
    color_vis = colors[label_map] 

    return label_map, color_vis

def Brightness(Img, num_clusters):
    """
    Generates a brightness map by clustering pixel intensities.
    Parameters:
        Img (np.ndarray): Input image (grayscale or color).
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        np.ndarray: Brightness map with clustered intensity levels.
    """
    # Convert to grayscale if color
    if len(Img.shape) == 3:
        Img_gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY if Img.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    else:
        Img_gray = Img

    # Normalize if uint8
    if Img_gray.dtype == np.uint8:
        Img_proc = Img_gray.astype(np.float32) / 255.0
    else:
        Img_proc = Img_gray.astype(np.float32)


    p, q = Img_proc.shape
    inp = Img_proc.reshape((p*q, 1))

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000, n_init='auto')
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    label_map = np.reshape(labels,(p,q))

    return label_map # Return the label map, visualization can be done with a grayscale cmap

def Color(Img, num_clusters):
    """
    Generates a color map by clustering pixel color values.
    Parameters:
        Img (np.ndarray): Input color image.
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        tuple: (2D label map, color visualization)
    """
    # Ensure Img is float for processing, typically [0,1] or [0,255]
    if Img.dtype == np.uint8:
        Img_proc = Img.astype(np.float32) / 255.0
    elif Img.max() > 1.0: # Assuming it might be float but not normalized
        Img_proc = Img.astype(np.float32) / 255.0
    else:
        Img_proc = Img.astype(np.float32)

    p,q,r = Img_proc.shape
    inp = np.reshape(Img_proc,((p*q),r))
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000, n_init='auto')
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    label_map = np.reshape(labels,(p,q))

    # Create deterministic color visualization
    colors = get_deterministic_colormap(num_clusters)
    # Directly map labels to colors
    color_vis = colors[label_map] 

    return label_map, color_vis


def chi_sqr_gradient(Img_map, bins, filter1, filter2):
    """
    Computes a gradient map using the Chi-squared distance between histograms.
    Parameters:
        Img_map (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins (distinct values in Img_map).
        filter1 (np.ndarray): First filter for gradient computation.
        filter2 (np.ndarray): Second filter for gradient computation.
    Returns:
        np.ndarray: Gradient map based on Chi-squared distance.
    """
    epsilon = 1e-10
    # Initialize chi_sqr_dist with zeros of the same shape as Img_map and float type
    chi_sqr_dist = np.zeros_like(Img_map, dtype=np.float32)
    for i in range(bins): # Iterate up to the number of bins (cluster labels)
        # Create a binary mask for the current bin/label
        img_mask = (Img_map == i).astype(np.float32) # Convert boolean to float (0.0 or 1.0)

        g = cv2.filter2D(img_mask, -1, filter1)
        h = cv2.filter2D(img_mask, -1, filter2)
        # Chi-squared calculation
        chi_sqr_dist += ((g-h)**2) / ((g+h) + epsilon)

    return chi_sqr_dist/2.0 # Divide by 2 as per formula


def Gradient(Img_map, bins, filter_bank): 
    """
    Computes a gradient map using a filter bank.
    Parameters:
        Img_map (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins.
        filter_bank (list): List of filters for gradient computation.
    Returns:
        tuple: (2D gradient map, color visualization)
    """
    # Initialize gradVar_list with the original map to simplify dstack later if needed,
    # but sum/mean of results from pairs of filters.
    gradient_responses = []

    num_filter_pairs = math.ceil(len(filter_bank)/2)
    if len(filter_bank) % 2 != 0:
        # Handle odd number of filters if necessary, e.g., by ignoring the last one or erroring.
        # For now, assuming filter_bank has an even number of elements for pairing.
        print("Warning: Odd number of filters in filter_bank for Gradient computation. Last filter ignored.")

    for N in range(num_filter_pairs):
        if (2*N + 1) < len(filter_bank): # Ensure pair exists
            g = chi_sqr_gradient(Img_map, bins, filter_bank[2*N], filter_bank[2*N+1])
            gradient_responses.append(g)

    if not gradient_responses: # If no gradients were computed (e.g., empty filter_bank)
        # Return zeros or handle error appropriately
        mean_gradient = np.zeros_like(Img_map, dtype=np.float32)
    else:
        # Stack responses and compute mean
        gradVar_stack = np.dstack(gradient_responses)
        mean_gradient = np.mean(gradVar_stack, axis=2)

    # Create color visualization using a colormap
    # Normalize mean_gradient to [0,1] for colormap
    normalized_mean = (mean_gradient - np.min(mean_gradient)) / (np.max(mean_gradient) - np.min(mean_gradient) + 1e-10)
    # Use jet colormap and remove alpha channel
    color_vis = plt.cm.jet(normalized_mean)[:, :, :3]  

    return mean_gradient, color_vis


def half_disc(radius):
    """
    Creates half-disc masks for gradient computation.
    Parameters:
        radius (int): Radius of the half-disc.
    Returns:
        tuple: Two binary masks (a, b) representing half-discs.
    """
    # Mask 'a' is the area outside the circle (complement of the disc)
    a = np.ones((2*radius+1, 2*radius+1), dtype=np.float32)
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask_circle = x*x + y*y <= radius**2
    # Set inside of circle to 0
    a[mask_circle] = 0 

    # Mask 'b' is one half of the disc (e.g., right half)
    b = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
    # Create a full disc first
    b[mask_circle] = 1
    # Then zero out one half (e.g., the left half and the center line x <= 0)
    b[x <= 0] = 0
    # a is complement, b is one half-disc.
    return a, b 
                
    # The original paper might use these differently, needs careful check against source.
    # For chi-squared gradients, we need two non-overlapping regions.
    # create two complementary half-discs directly.

def create_half_disc_pair(radius):
    """
    Creates a pair of complementary half-disc masks.
    Parameters:
        radius (int): Radius of the half-disc.
    Returns:
        tuple: (mask_left_half, mask_right_half)
    """
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    # Full circular mask
    circle_mask = x*x + y*y <= radius**2

    # Left half-disc (x < 0 within the circle)
    mask_left_half = np.zeros((diameter, diameter), dtype=np.float32)
    mask_left_half[(x < 0) & circle_mask] = 1

    # Right half-disc (x > 0 within the circle)
    mask_right_half = np.zeros((diameter, diameter), dtype=np.float32)
    mask_right_half[(x > 0) & circle_mask] = 1
    
    # Normalize masks (sum of elements in each mask is 1)
    # important for some gradient computations.
    sum_left = np.sum(mask_left_half)
    if sum_left > 0:
        mask_left_half /= sum_left
    
    sum_right = np.sum(mask_right_half)
    if sum_right > 0:
        mask_right_half /= sum_right
        
    return mask_left_half, mask_right_half


def disc_masks(radii_list, orients):
    """
    Generates a set of rotated half-disc masks.
    Parameters:
        radii_list (list): List of radii for the half-discs.
        orients (int): Number of orientations for the half-discs.
    Returns:
        list: A list of rotated half-disc masks (pairs of filters).
    """
    filter_pairs = [] # Store pairs of (left_half, right_half)
    angles = np.linspace(0, 360, orients, endpoint=False)
    for rad in radii_list:
        left_half, right_half = create_half_disc_pair(rad)
        for ang_deg in angles:
            # Rotate both half-discs
            # cval=0 ensures parts rotated out of bounds are 0, not 1
            # mode='constant' fills with cval
            # order=0 for nearest neighbor, preserving binary nature if not already float
            rotated_left = skimage.transform.rotate(left_half, ang_deg, resize=False, cval=0, mode='constant', preserve_range=True, order=0)
            rotated_right = skimage.transform.rotate(right_half, ang_deg, resize=False, cval=0, mode='constant', preserve_range=True, order=0)
            
            # Normalize again after rotation, as rotation might slightly change sums due to interpolation/aliasing
            # if order > 0. With order=0, sum should be preserved if resize=False.
            sum_rot_left = np.sum(rotated_left)
            if sum_rot_left > 0: rotated_left /= sum_rot_left
            
            sum_rot_right = np.sum(rotated_right)
            if sum_rot_right > 0: rotated_right /= sum_rot_right

            filter_pairs.append(rotated_left)
            filter_pairs.append(rotated_right)

    return filter_pairs


def normalize_image(img):
    """
    Normalize image values to [0,1] range.
    Handles float and integer images.
    """
    if img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32) 

    img_min = np.min(img_float)
    img_max = np.max(img_float)
    if img_max > img_min:
        return (img_float - img_min) / (img_max - img_min)
    elif img_max == img_min and img_min == 0: # All zeros
        return img_float
    elif img_max == img_min and img_min != 0: # All same non-zero value, normalize to 1
        return img_float / img_min 
    
    return img_float # Should not happen if max > min, but as a fallback

def process_image(image_path, save_dir=None):
    """
    Process a single image through the edge detection pipeline.

    Args:
        image_path (str): Path to the input image
        save_dir (str, optional): Directory to save results. If None, results won't be saved.

    Returns:
        dict: Dictionary containing all intermediate and final results
    """
    # Load image using OpenCV for better format handling, then convert to RGB for Matplotlib consistency
    # robust image loading, correctly handling EXIF orientation and color spaces
    try:
        pil_image = Image.open(image_path)
        img = np.array(pil_image.convert('RGB')) 
    except Exception as e:
        print(f"Pillow failed to open image: {e}. Falling back to OpenCV.")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read the image from path: {image_path}. Please check if the file exists and is a valid image.")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    original_size = img.shape[:2]  # Store original size (height, width)

    # Ensure image is 3-channel RGB
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # If RGBA, Remove alpha channel
    elif len(img.shape) == 3 and img.shape[2] == 4:  
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) 

    # Normalize image to [0, 1] float range if it's uint8
    if img.dtype == np.uint8:
        img_proc = img.astype(np.float32) / 255.0
    else:
        img_proc = img.copy() # Assume it's already float, possibly [0,1] or other range

    # Resize for processing if image is too large
    max_dimension = 600 # Reduced max_dimension for faster processing
    height, width = img_proc.shape[:2]
    resized_for_processing = False
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_proc_resized = cv2.resize(img_proc, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {width}x{height} to {new_width}x{new_height} for processing")
        resized_for_processing = True
    else:
        img_proc_resized = img_proc.copy()


    # Generate filter banks (consider caching these if called repeatedly in an app)
    # Adjusted parameters for potentially faster processing. Original: kernelsize=49
    LM_KERNEL_SIZE = 35 # Reduced from 49
    DOG_SIZE = 35       # Reduced from 49
    filter_bank1 = LM_filters(kernelsize=LM_KERNEL_SIZE, sigma_scales=3, num_orientations=6, nrotinv=12)
    filter_bank2 = Oriented_DOG(sigma=[3,5,7], orient=8, size=DOG_SIZE) # Reduced sigma, orient, size
    filter_bank3 = Gabor_filter(sigma_list=[3,5,7], theta=0, Lambda=4.0, psi=0, gamma=0.70, num_filters=8) # Reduced sigma, num_filters
    half_disc_masks = disc_masks(radii_list=[3,5,10], orients=6) # Reduced radii, orients

    # Generate texture map
    # Pass the resized image to Texton
    T_labels, T_color_vis = Texton(img_proc_resized, filter_bank1, filter_bank2, filter_bank3, num_clusters=32) # Reduced clusters

    # Generate brightness map (use the resized image)
    # Brightness function expects RGB or Grayscale. If RGB, it converts to gray.
    B_labels = Brightness(img_proc_resized, num_clusters=8) # Reduced clusters
    B_color_vis = plt.cm.gray(normalize_image(B_labels))[:,:,:3] # Grayscale visualization

    # Generate color map (use the resized image)
    C_labels, C_color_vis = Color(img_proc_resized, num_clusters=8) # Reduced clusters

    # Generate gradients using the label maps
    Tg_map, Tg_color_vis = Gradient(T_labels, 32, half_disc_masks) # bins = num_clusters for Texton
    Bg_map, Bg_color_vis = Gradient(B_labels, 8, half_disc_masks)  # bins = num_clusters for Brightness
    Cg_map, Cg_color_vis = Gradient(C_labels, 8, half_disc_masks)  # bins = num_clusters for Color

    # Normalize all gradient maps before combining
    Tg_norm = normalize_image(Tg_map)
    Bg_norm = normalize_image(Bg_map)
    Cg_norm = normalize_image(Cg_map)

    # Generate Sobel and Canny baselines using the resized, processed image
    # Convert to grayscale for Sobel/Canny, ensure it's uint8 [0,255]
    gray_img_proc = cv2.cvtColor(img_proc_resized, cv2.COLOR_RGB2GRAY)

    #convert grayscale into uint8 [0,255] format 
    if gray_img_proc.dtype != np.uint8:
        gray_img_for_cv = (normalize_image(gray_img_proc) * 255).astype(np.uint8)
    else:
        gray_img_for_cv = gray_img_proc


    sobel_x = cv2.Sobel(gray_img_for_cv, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img_for_cv, cv2.CV_64F, 0, 1, ksize=3)
    sobel_baseline = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_baseline_norm = normalize_image(sobel_baseline)

    # Canny expects uint8 image
    canny_baseline = cv2.Canny(gray_img_for_cv, 100, 200)
    canny_baseline_norm = normalize_image(canny_baseline.astype(np.float32)) # Canny is binary, normalize to 0 or 1

    # Combine responses
    alpha, beta, gamma = 0.33, 0.33, 0.34
    combined_gradient = alpha*Tg_norm + beta*Bg_norm + gamma*Cg_norm
    combined_gradient_norm = normalize_image(combined_gradient)

    # Generate final output: multiply with weighted Canny and Sobel
    # Ensure all components are in [0,1] range before multiplication
    pblite_output = combined_gradient_norm * (0.50*canny_baseline_norm + 0.50*sobel_baseline_norm)
    pblite_output_norm = normalize_image(pblite_output)


    # Resize final output and intermediate maps back to original size if image was resized for processing
    final_results = {
        'texture_map': T_color_vis,
        'brightness_map': B_color_vis,
        'color_map': C_color_vis,
        'texture_gradient': Tg_color_vis,
        'brightness_gradient': Bg_color_vis,
        'color_gradient': Cg_color_vis,
        'sobel_baseline': sobel_baseline_norm, # Return normalized
        'canny_baseline': canny_baseline_norm, # Return normalized
        'final_output': pblite_output_norm
    }

    if resized_for_processing:
        for key, value_img in final_results.items():
            if value_img.ndim == 2: # Grayscale images
                final_results[key] = cv2.resize(value_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            elif value_img.ndim == 3: # Color images
                 final_results[key] = cv2.resize(value_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            # Ensure resized images maintain [0,1] range if they were normalized
            final_results[key] = normalize_image(final_results[key])


    # Save results if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Matplotlib imsave expects float [0,1] or uint8 [0,255]
        # Our visualizations are generally [0,1] float.
        # For binary/label maps like B_labels, specify cmap.
        plt.imsave(os.path.join(save_dir, 'texture_map_vis.png'), final_results['texture_map'])
        plt.imsave(os.path.join(save_dir, 'brightness_map_vis.png'), final_results['brightness_map'], cmap='gray') # Save the visual
        plt.imsave(os.path.join(save_dir, 'color_map_vis.png'), final_results['color_map'])
        plt.imsave(os.path.join(save_dir, 'texture_gradient_vis.png'), final_results['texture_gradient'])
        plt.imsave(os.path.join(save_dir, 'brightness_gradient_vis.png'), final_results['brightness_gradient'])
        plt.imsave(os.path.join(save_dir, 'color_gradient_vis.png'), final_results['color_gradient'])
        plt.imsave(os.path.join(save_dir, 'final_pb_lite_output.png'), final_results['final_output'], cmap='gray')

    return final_results

def process_image_from_memory(pil_image):
    """
    Process a PIL Image object through the edge detection pipeline.
    
    This is a wrapper around process_image that works with in-memory images
    instead of file paths, making it suitable for API use.

    Args:
        pil_image (PIL.Image): Input PIL Image object

    Returns:
        dict: Dictionary containing all intermediate and final results as numpy arrays
    """
    # Convert PIL image to numpy array in RGB format
    img = np.array(pil_image.convert('RGB'))
    
    original_size = img.shape[:2]  # Store original size (height, width)

    # Ensure image is 3-channel RGB
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # If RGBA, Remove alpha channel
    elif len(img.shape) == 3 and img.shape[2] == 4:  
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) 

    # Normalize image to [0, 1] float range if it's uint8
    if img.dtype == np.uint8:
        img_proc = img.astype(np.float32) / 255.0
    else:
        img_proc = img.copy() # Assume it's already float, possibly [0,1] or other range

    # Resize for processing if image is too large
    max_dimension = 600 # Reduced max_dimension for faster processing
    height, width = img_proc.shape[:2]
    resized_for_processing = False
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_proc_resized = cv2.resize(img_proc, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {width}x{height} to {new_width}x{new_height} for processing")
        resized_for_processing = True
    else:
        img_proc_resized = img_proc.copy()

    # Generate filter banks (consider caching these if called repeatedly in an app)
    # Adjusted parameters for potentially faster processing. Original: kernelsize=49
    LM_KERNEL_SIZE = 35 # Reduced from 49
    DOG_SIZE = 35       # Reduced from 49
    filter_bank1 = LM_filters(kernelsize=LM_KERNEL_SIZE, sigma_scales=3, num_orientations=6, nrotinv=12)
    filter_bank2 = Oriented_DOG(sigma=[3,5,7], orient=8, size=DOG_SIZE) # Reduced sigma, orient, size
    filter_bank3 = Gabor_filter(sigma_list=[3,5,7], theta=0, Lambda=4.0, psi=0, gamma=0.70, num_filters=8) # Reduced sigma, num_filters
    half_disc_masks = disc_masks(radii_list=[3,5,10], orients=6) # Reduced radii, orients

    # Generate texture map
    # Pass the resized image to Texton
    T_labels, T_color_vis = Texton(img_proc_resized, filter_bank1, filter_bank2, filter_bank3, num_clusters=32) # Reduced clusters

    # Generate brightness map (use the resized image)
    # Brightness function expects RGB or Grayscale. If RGB, it converts to gray.
    B_labels = Brightness(img_proc_resized, num_clusters=8) # Reduced clusters
    B_color_vis = plt.cm.gray(normalize_image(B_labels))[:,:,:3] # Grayscale visualization

    # Generate color map (use the resized image)
    C_labels, C_color_vis = Color(img_proc_resized, num_clusters=8) # Reduced clusters

    # Generate gradients using the label maps
    Tg_map, Tg_color_vis = Gradient(T_labels, 32, half_disc_masks) # bins = num_clusters for Texton
    Bg_map, Bg_color_vis = Gradient(B_labels, 8, half_disc_masks)  # bins = num_clusters for Brightness
    Cg_map, Cg_color_vis = Gradient(C_labels, 8, half_disc_masks)  # bins = num_clusters for Color

    # Normalize all gradient maps before combining
    Tg_norm = normalize_image(Tg_map)
    Bg_norm = normalize_image(Bg_map)
    Cg_norm = normalize_image(Cg_map)

    # Generate Sobel and Canny baselines using the resized, processed image
    # Convert to grayscale for Sobel/Canny, ensure it's uint8 [0,255]
    gray_img_proc = cv2.cvtColor(img_proc_resized, cv2.COLOR_RGB2GRAY)

    #convert grayscale into uint8 [0,255] format 
    if gray_img_proc.dtype != np.uint8:
        gray_img_for_cv = (normalize_image(gray_img_proc) * 255).astype(np.uint8)
    else:
        gray_img_for_cv = gray_img_proc

    sobel_x = cv2.Sobel(gray_img_for_cv, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img_for_cv, cv2.CV_64F, 0, 1, ksize=3)
    sobel_baseline = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_baseline_norm = normalize_image(sobel_baseline)

    # Canny expects uint8 image
    canny_baseline = cv2.Canny(gray_img_for_cv, 100, 200)
    canny_baseline_norm = normalize_image(canny_baseline.astype(np.float32)) # Canny is binary, normalize to 0 or 1

    # Combine responses
    alpha, beta, gamma = 0.33, 0.33, 0.34
    combined_gradient = alpha*Tg_norm + beta*Bg_norm + gamma*Cg_norm
    combined_gradient_norm = normalize_image(combined_gradient)

    # Generate final output: multiply with weighted Canny and Sobel
    # Ensure all components are in [0,1] range before multiplication
    pblite_output = combined_gradient_norm * (0.50*canny_baseline_norm + 0.50*sobel_baseline_norm)
    pblite_output_norm = normalize_image(pblite_output)

    # Resize final output and intermediate maps back to original size if image was resized for processing
    final_results = {
        'texture_map': T_color_vis,
        'brightness_map': B_color_vis,
        'color_map': C_color_vis,
        'texture_gradient': Tg_color_vis,
        'brightness_gradient': Bg_color_vis,
        'color_gradient': Cg_color_vis,
        'sobel_baseline': sobel_baseline_norm, # Return normalized
        'canny_baseline': canny_baseline_norm, # Return normalized
        'final_output': pblite_output_norm
    }

    if resized_for_processing:
        for key, value_img in final_results.items():
            if value_img.ndim == 2: # Grayscale images
                final_results[key] = cv2.resize(value_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            elif value_img.ndim == 3: # Color images
                 final_results[key] = cv2.resize(value_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            # Ensure resized images maintain [0,1] range if they were normalized
            final_results[key] = normalize_image(final_results[key])

    return final_results
