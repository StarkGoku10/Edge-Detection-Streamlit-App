#!/usr/bin/env python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import skimage.transform
import scipy.stats as st
from sklearn.cluster import MiniBatchKMeans

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
    im_max = np.max(np.abs(image)) + 1e-10
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
    for scale in sigma:
        orients = np.linspace(0,360,orient, endpoint=False)
        kernel = Gaussiankernel2D(size,scale)
        sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=3, borderType=border)
        smax = np.max(np.abs(sobelx64f)) + 1e-10
        sobelx64f /= smax
        for eachOrient in orients:
            rotated = skimage.transform.rotate(sobelx64f, eachOrient, mode='reflect')
            rmax = np.max(np.abs(rotated)) + 1e-10
            rotated /= rmax
            kernels.append(rotated)
    return kernels

def LM_filters(kernelsize, sigma, num_orientations, nrotinv):
    """
    Generate Leung-Malik (LM) filter bank.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (int): Number of scales.
        num_orientations (int): Number of orientations.
        nrotinv (int): Number of rotationally invariant filters.
    Returns:
        np.ndarray: Array of LM filters.
    """
    scalex = np.sqrt(2) * np.arange(1,sigma+1)
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
        for orient in range(num_orientations):
            angle = (math.pi * orient)/num_orientations
            c = np.cos(angle)
            s = np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
            rotpts = rot_mat.dot(orgpts)
            F[:,:,count] = makefilter(scaleVal, 0, 1, rotpts, kernelsize)
            F[:,:,count+nedge] = makefilter(scaleVal, 0, 2, rotpts, kernelsize)
            count = count + 1
    count = nbar+nedge
    sigma_vals = np.sqrt(2) * np.array([1,2,3,4], dtype=np.float32)

    for i in range(len(sigma_vals)):
        g2d = Gaussian2D(kernelsize, sigma_vals[i])
        g2d /= (np.max(np.abs(g2d)) + 1e-10)
        F[:, :, count] = g2d
        count += 1

    for i in range(len(sigma_vals)):
        log_2d = lap_gaussian2D(kernelsize, sigma_vals[i])
        F[:, :, count] = log_2d
        count += 1

    for i in range(len(sigma_vals)):
        log_2d = lap_gaussian2D(kernelsize, 3*sigma_vals[i])
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
    x_max = int(np.ceil(max(abs(nstds*sigma_x*np.cos(theta)),
                            abs(nstds*sigma_y*np.sin(theta)))))
    y_max = int(np.ceil(max(abs(nstds*sigma_x*np.sin(theta)),
                            abs(nstds*sigma_y*np.cos(theta)))))
    x_min, y_min = -x_max, -y_max
    (y, x) = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(
        -0.5 * (x_theta**2 / (sigma_x**2) + y_theta**2 / (sigma_y**2))
    ) * np.cos(2*np.pi / Lambda * x_theta + psi)
    denom = np.sqrt((gb**2).sum()) + 1e-10
    gb = gb / denom
    return gb

def Gabor_filter(sigma, theta, Lambda, psi, gamma, num_filters):
    """
    Creates a set of Gabor filters with varying orientations.
    Parameters:
        sigma (list): List of standard deviations for the Gaussian envelope.
        theta (float): Fixed orientation for the base Gabor filter.
        Lambda (float): Wavelength of the sinusoidal carrier.
        psi (float): Phase offset of the sinusoidal carrier.
        gamma (float): Spatial aspect ratio (ellipticity) of the Gabor filter.
        num_filters (int): Number of filters to generate with different orientations.
    Returns:
        list: A list of 2D Gabor filter kernels.
    """
    g = []
    ang = np.linspace(0, 360, num_filters, endpoint=False)
    for s in sigma:
        base = gabor_fn(s, theta, Lambda, psi, gamma)
        for a in ang:
            image = skimage.transform.rotate(base,a,mode='reflect')
            imax = np.max(np.abs(image)) + 1e-10
            image /= imax
            g.append(image)
    return g

def textonmap_DOG(Img, filter_bank):
    """
    Generates a texton map using a Difference of Gaussian (DoG) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (list): List of DoG filter kernels.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    tex_map = np.array(Img)
    for i in range(len(filter_bank)):
        out = cv2.filter2D(Img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

def textonmap_LM(Img, filter_bank):
    """
    Generates a texton map using an LM (Leung-Malik) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (np.ndarray): A 3D array representing the LM filter bank.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    tex_map = np.array(Img)
    _,_,num_filters = filter_bank.shape
    for i in range(num_filters):
        out = cv2.filter2D(Img,-1,filter_bank[:,:,i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

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
    p,q,_ = img.shape
    weights = [0.3,0.3,0.4] 
    
    tex_map_DOG = textonmap_DOG(img, filter_bank2) * weights[0]
    tex_map_LM = textonmap_LM(img, filter_bank1) * weights[1]
    tex_map_Gabor = textonmap_DOG(img, filter_bank3) * weights[2]
    
    combined = np.dstack((
        tex_map_DOG[:, :, 1:],
        tex_map_LM[:, :, 1:],
        tex_map_Gabor[:, :, 1:]
    ))
    
    m,n,r = combined.shape
    inp = np.reshape(combined,((p*q),r))
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(m,n))
    
    # Create color visualization
    color_vis = np.zeros((m,n,3))
    for i in range(num_clusters):
        mask = (l == i)
        color_vis[mask] = np.random.rand(3)
    
    return l, color_vis

def Brightness(Img, num_clusters):
    """
    Generates a brightness map by clustering pixel intensities.
    Parameters:
        Img (np.ndarray): Input image (grayscale or color).
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        np.ndarray: Brightness map with clustered intensity levels.
    """
    # Handle both grayscale and color images
    if len(Img.shape) == 2:  # Grayscale image
        p, q = Img.shape
        inp = Img.reshape((p*q, 1))
    else:  # Color image
        p, q, r = Img.shape
        inp = Img.reshape((p*q, r))
    
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    return l

def Color(Img, num_clusters):
    """
    Generates a color map by clustering pixel color values.
    Parameters:
        Img (np.ndarray): Input color image.
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        tuple: (2D label map, color visualization)
    """  
    p,q,r = Img.shape
    inp = np.reshape(Img,((p*q),r))
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    
    # Create color visualization
    color_vis = np.zeros((p,q,3))
    for i in range(num_clusters):
        mask = (l == i)
        color_vis[mask] = np.random.rand(3)
    
    return l, color_vis

def chi_sqr_gradient(Img, bins, filter1, filter2):
    """
    Computes a gradient map using the Chi-squared distance between histograms.
    Parameters:
        Img (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins.
        filter1 (np.ndarray): First filter for gradient computation.
        filter2 (np.ndarray): Second filter for gradient computation.
    Returns:
        np.ndarray: Gradient map based on Chi-squared distance.
    """
    epsilon = 1e-10
    chi_sqr_dist = Img*0  
    for i in range(bins):
        img_mask = np.ma.masked_where(Img == i, Img).mask.astype(np.int32)
        g = cv2.filter2D(img_mask, -1, filter1)
        h = cv2.filter2D(img_mask, -1, filter2)
        chi_sqr_dist = chi_sqr_dist + ((g-h)**2) / ((g+h) + epsilon)
    return chi_sqr_dist/2

def Gradient(Img, bins, filter_bank):
    """
    Computes a gradient map using a filter bank.
    Parameters:
        Img (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins.
        filter_bank (list): List of filters for gradient computation.
    Returns:
        tuple: (2D gradient map, color visualization)
    """    
    gradVar = Img
    for N in range(math.ceil(len(filter_bank)/2)):
        g = chi_sqr_gradient(Img, bins, filter_bank[2*N], filter_bank[2*N+1])
        gradVar = np.dstack((gradVar, g))
    mean = np.mean(gradVar, axis=2)
    
    # Create color visualization using a colormap
    normalized_mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean) + 1e-10)
    color_vis = plt.cm.jet(normalized_mean)[:, :, :3]  # Use jet colormap and remove alpha channel
    
    return mean, color_vis

def half_disc(radius):
    """
    Creates half-disc masks for gradient computation.
    Parameters:
        radius (int): Radius of the half-disc.
    Returns:
        tuple: Two binary masks (a, b) representing half-discs.
    """
    a = np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b = np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1
    q = y>-radius-1
    mask3 = p&q
    b[mask3] = 0
    return a, b

def disc_masks(sigma, orients):
    """
    Generates a set of rotated half-disc masks.
    Parameters:
        sigma (list): List of radii for the half-discs.
        orients (int): Number of orientations for the half-discs.
    Returns:
        list: A list of rotated half-disc masks.
    """    
    flt = []
    angles = np.linspace(0,360,orients, endpoint=False)
    for rad in sigma:
        a,b = half_disc(radius=rad)
        for ang in angles:
            c1 = skimage.transform.rotate(b,ang,cval=1, mode='constant')
            z1 = np.logical_or(a,c1).astype(np.int32)
            b2 = np.flip(b,1)
            c2 = skimage.transform.rotate(b2,ang,cval=1, mode='constant')
            z2 = np.logical_or(a,c2).astype(np.int32)
            flt.append(z1)
            flt.append(z2)
    return flt

def normalize_image(img):
    """
    Normalize image values to [0,1] range.
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def process_image(image_path, save_dir=None):
    """
    Process a single image through the edge detection pipeline.
    
    Args:
        image_path (str): Path to the input image
        save_dir (str, optional): Directory to save results. If None, results won't be saved.
        
    Returns:
        dict: Dictionary containing all intermediate and final results
    """
    img = plt.imread(image_path)
    original_size = img.shape[:2]  # Store original size
    
    # Validate image
    if img is None:
        raise ValueError("Could not read the image. Please check if the file exists and is a valid image.")
    
    # Ensure image is in RGB format
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # If RGBA
        img = img[:, :, :3]  # Remove alpha channel
    
    # Resize for processing if image is too large
    max_dimension = 800
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {width}x{height} to {new_width}x{new_height} for processing")
    
    img_col = img.copy()  # Keep color version
    
    # Generate filter banks
    filter_bank1 = LM_filters(kernelsize=49, sigma=3, num_orientations=6, nrotinv=12)
    filter_bank2 = Oriented_DOG(sigma=[5,8,11], orient=15, size=49)
    filter_bank3 = Gabor_filter(sigma=[5,8,11], theta=0, Lambda=4.0, psi=0, gamma=0.70, num_filters=15)
    half_disc_masks = disc_masks([5,7,16], 8)
    
    # Generate texture map
    T, T_color = Texton(img_col, filter_bank1, filter_bank2, filter_bank3, num_clusters=64)
    
    # Generate brightness map
    B = Brightness(img, num_clusters=16)
    
    # Generate color map
    C, C_color = Color(img_col, num_clusters=16)
    
    # Generate gradients
    Tg, Tg_color = Gradient(T, 64, half_disc_masks)
    Bg, Bg_color = Gradient(B, 16, half_disc_masks)
    Cg, Cg_color = Gradient(C, 16, half_disc_masks)
    
    # Normalize the brightness gradient
    Bg = normalize_image(Bg)
    
    # Generate Sobel and Canny baselines
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_baseline = np.sqrt(sobel_x**2 + sobel_y**2)
    canny_baseline = cv2.Canny(gray_img, 100, 200)
    
    # Combine responses
    alpha, beta, gamma = 0.33, 0.33, 0.34
    combined_gradient = alpha*Tg + beta*Bg + gamma*Cg
    
    # Generate final output
    pblite_output = combined_gradient * (0.50*canny_baseline + 0.50*sobel_baseline)
    
    # Normalize only the final output
    pblite_output = normalize_image(pblite_output)
    
    # Resize only the final Pb-lite output if image was resized
    if img.shape[:2] != original_size:
        pblite_output = cv2.resize(pblite_output, (original_size[1], original_size[0]), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # Save results if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.imsave(os.path.join(save_dir, 'texture_map.png'), T_color)
        plt.imsave(os.path.join(save_dir, 'brightness_map.png'), B, cmap='binary')
        plt.imsave(os.path.join(save_dir, 'color_map.png'), C_color)
        plt.imsave(os.path.join(save_dir, 'texture_gradient.png'), Tg_color)
        plt.imsave(os.path.join(save_dir, 'brightness_gradient.png'), Bg, cmap='binary')
        plt.imsave(os.path.join(save_dir, 'color_gradient.png'), Cg_color)
        plt.imsave(os.path.join(save_dir, 'final_output.png'), pblite_output, cmap='gray')
    
    return {
        'texture_map': T_color,
        'brightness_map': B,
        'color_map': C_color,
        'texture_gradient': Tg_color,
        'brightness_gradient': Bg,
        'color_gradient': Cg_color,
        'sobel_baseline': sobel_baseline,
        'canny_baseline': canny_baseline,
        'final_output': pblite_output
    } 