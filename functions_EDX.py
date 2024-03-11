# Functions for EDX
import numpy as np
from sklearn.decomposition import PCA
import sys
import time
from ipywidgets import interactive
import matplotlib.pyplot as plt
from scipy.optimize import nnls 
from scipy.stats import zscore
from datetime import datetime
import seaborn as sns
from skimage.feature import peak_local_max
from matplotlib import cm
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
from VCA import *
import os
from scipy import signal
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from scipy.io import savemat,loadmat
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, widgets, Button, Output
import pacmap
from annoy import AnnoyIndex
import glob



def reconstruct_manual(X,pca,which_components):
    comps = pca.components_
    mu = pca.mean_
    X_transformed = pca.transform(X)
    return np.dot(X_transformed[:,which_components],comps[which_components,:])+mu


def GaussFilter(im,apply=True,sigma = 2, size=3):
    if apply:
        kernel = np.ones((size,size),np.float32)/(size*size)
        im_filtered = cv.GaussianBlur(im,(size,size),sigmaX = sigma, sigmaY= sigma, borderType =cv.BORDER_DEFAULT)
    else:
        im_filtered= im
    return im_filtered

def MeanFilter(im,apply=True,size=3):
    if apply:
        kernel = np.ones((size,size),np.float32)/(size*size)
        im_filtered = cv.filter2D(im,-1,kernel) 
    else:
        im_filtered= im
    return im_filtered

def GaussFilterCube(spectrum, sigma = 2, size=3):
    spectrum_filtered = np.zeros(spectrum.shape)
    for i in range(spectrum.shape[2]): 
        spectrum_filtered[:,:,i] = GaussFilter(spectrum[:,:,i],apply=True,sigma=sigma,size=size)
    return spectrum_filtered

def MeanFilterCube(spectrum, sigma = 2, size=3):
    spectrum_filtered = np.zeros(spectrum.shape)
    for i in range(spectrum.shape[2]): 
        spectrum_filtered[:,:,i] = MeanFilter(spectrum[:,:,i],apply=True,size=size)
    return spectrum_filtered


def discrete_matshow(data,cmap='RdBu'):
    # get discrete colormap
    cmap = plt.get_cmap(cmap, np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))

def median_blur(im_in,size=0):
    if size==0:
        im_out = im_in
    else: 
        im_out = cv.medianBlur(np.uint8(im_in),size)
    return im_out

def rebin_spectrum(spectrum,bins=1024):
    x,y,z = spectrum.shape
    spectrum = np.reshape(spectrum,(x,y,bins,int(z/bins)))
    return np.sum(spectrum,axis=-1)

def rebin_XY(img,bins=1024):
    x,y = img.shape
    img= np.reshape(img,(bins,int(x/bins),bins,int(y/bins)))
    return img.mean(axis=-1).mean(axis=1)

def rebin_spectrumXY(spectrum,bins=1024):
    x,y,z = spectrum.shape
    spectrum = np.reshape(spectrum,(bins,int(x/bins),bins,int(y/bins),z))
    spectrum = np.mean(spectrum,axis=-2)
    return spectrum.mean(axis=1)

#def rebin_spectrumXY(spectrum,bins=1024):    # Too slow!
#    x,y,z = spectrum.shape
#    new_spectrum = np.zeros((bins, bins, z))
#    for k in range(z):
#        new_spectrum[:,:,k] = rebin_XY(spectrum[:,:,k])
#        print(k)
#    return new_spectrum

def rebin_energies(energies, bins):
    z = energies.shape[0]
    energies = energies.reshape((bins, int(z/bins))).mean(axis=1)
    return energies

def clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    img = img.astype('uint8')
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

def MinMaxHSI(spectrum_2D):
    for j in range(spectrum_2D.shape[1]):
        spectrum_2D[:,j] = NormalizeData(spectrum_2D[:,j])
    return spectrum_2D

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def make_ome_rgb(spectrum_rgb,outpath,lvl=4,downsample_factor=2,pixel_size=2.5,tile_size=32,ch_type='PC'):
    with tif.TiffWriter(outpath, bigtiff=True) as tf:
        data = spectrum_rgb
        data = np.swapaxes(data,2,0)
        data = np.swapaxes(data,1,2)
        options = dict(photometric='rgb', tile=(tile_size, tile_size), compression='None',metadata={'axes': 'CYX'})
        tf.write(data, subifds=lvl, **options)
        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolutions
        tf.write(data[:, ::2, ::2], subfiletype=1, **options)
        tf.write(data[:, ::4, ::4], subfiletype=1, **options)
        tf.write(data[:, ::8, ::8], subfiletype=1, **options)


def make_ome(haadf, spectrum_reduced,outpath,lvl=4,downsample_factor=2,pixel_size=2.5,tile_size=32,ch_type='PC'):
    compression = None
    subsample_size = haadf.shape[0]
    haadf_reshape = haadf.reshape((subsample_size,subsample_size,1))
    image_temp = np.concatenate((haadf_reshape,spectrum_reduced),axis=2)
    image_temp = np.swapaxes(image_temp,2,0)
    image_temp = np.swapaxes(image_temp,1,2)
    print(image_temp.shape)

    image = np.zeros(image_temp.shape,dtype='uint8')
    for i in range(image.shape[0]):
        image[i,:,:] = normalize8(image_temp[i,:,:])


    ch_names = ['Haadf']
    for j in range(spectrum_reduced.shape[2]):
        ch_names.append('%s_%02d' % (ch_type,(j+1)))
        
    with tif.TiffWriter(outpath, bigtiff=True) as tf:
        options = {'tile': (tile_size, tile_size),
                           'compression': compression,
                           'metadata':{'PhysicalSizeX': pixel_size*1e9, 'PhysicalSizeXUnit': 'nm',
                                       'PhysicalSizeY': pixel_size*1e9, 'PhysicalSizeYUnit': 'nm',
                                       'axes': 'CYX','Description':"Who dis",
                                       'AcquisitionDate':'Now',
                                       'Name':"oh nana",
                                       'Channel': {'Name':ch_names}}}
        tf.write(image, subifds=lvl, **options)
     
        image2 = image
        
        for i in range(lvl):
            idx = downsample_factor**(i+1)
            tf.write(image2[:,::idx, ::idx], subfiletype=1, **options)


def normalize8(I,normalize_by=None):
  if normalize_by is None:  
    mn = I.min()
    mx = I.max()
  else:
    mn = normalize_by.min()
    mx = normalize_by.max()

  mx -= mn
  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

# use this one if you want to normalize by another array
def normalize88(I,normalizer=None):
    if normalizer is None:  
        mn = I.min()
        mx = I.max()
        #print(mx)
    else:
        mn = normalizer.min()
        mx = normalizer.max()
        #print(mx)

    mx -= mn
    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8),mn,mx


# NNLS to FCLS
def nnls2fcls(End_maps):
    EM,x,y,files = End_maps.shape
    for i in range(x):
        for j in range(y):
            for f in range(files):
                End_maps[:,i,j,f] = End_maps[:,i,j,f]/np.sum(End_maps[:,i,j,f])
    return End_maps

def nnls_maxcf(End_maps):
    return End_maps/End_maps.max()


def mosaic_edx(main_folder='/data/p276451/EDX/',rows=[0,2],cols=[0,2],base_dims=[1024,1024,256],hash_sample=None,crop_neg=96):
    haadf = np.zeros((base_dims[0]*(rows[1]-rows[0]),base_dims[1]*(cols[1]-cols[0])))

    if hash_sample is None:
        spectrum = np.zeros((base_dims[0]*(rows[1]-rows[0]),base_dims[1]*(cols[1]-cols[0]),base_dims[2]))
    else:
        hash_dim = len(np.arange(1024)[::9])
        spectrum = np.zeros((hash_dim*(rows[1]-rows[0]),hash_dim*(cols[1]-cols[0]),base_dims[2]))

    for i in range(rows[0],rows[1]):
        for j in range(cols[0],cols[1]):
            file_path = os.path.join(main_folder,'npz_files/row%02d_col%02d.npz' % (i,j))
            loaded_file = np.load(file_path)
            print(file_path)
            haadf_temp = loaded_file['haadf']
            spectrum_temp = rebin_spectrum(loaded_file['spectrum'][:,:,crop_neg:],bins=base_dims[2])

            if hash_sample is not None:
                spectrum_temp = spectrum_temp[::hash_sample,::hash_sample,:]
            
            haadf[base_dims[0]*(i):base_dims[0]*(i)+base_dims[0],
                  base_dims[1]*(j):base_dims[0]*(j)+base_dims[1]] = haadf_temp

            if hash_sample is None:
                spectrum[base_dims[0]*(i):base_dims[0]*(i)+base_dims[0],
                      base_dims[1]*(j):base_dims[0]*(j)+base_dims[1],:] = spectrum_temp
            else:
                spectrum[hash_dim*(i):hash_dim*(i)+hash_dim,
                      hash_dim*(j):hash_dim*(j)+hash_dim,:] = spectrum_temp

    return haadf,spectrum

def show_anns(anns,display=False,area_thresh=1024**2):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        if ann['area'] < area_thresh:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            #color_mask = np.asarray([1,0,0,0.35])
            img[m] = color_mask
    
    if display:
        ax.imshow(img)
    else:
        return img
    
    
def show_anns_EDX(anns,abundance_tile,colors,display=False,alpha=0.35,area_thresh=1024**2,min_purity=0):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    img_clr_idx = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))*-1
    
    for ann in sorted_anns:
        if ann['area'] < area_thresh:
            m = ann['segmentation']
            tmp_img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            tmp_img[m] = 1
            tmp_abundance_masked = tmp_img*abundance_tile
            temp_sum = np.sum(np.sum(tmp_abundance_masked,axis=1),axis=1)

            color_idx = np.argmax(temp_sum)
            if (np.max(temp_sum)/ann['area']/255)>=min_purity:
                img_clr_idx[m] = color_idx
                color_mask = np.concatenate([colors[color_idx], [alpha]])
                img[m] = color_mask
    
    if display:
        ax.imshow(img)
    else:
        return img,img_clr_idx



def overlap_corr(tile_idx,xy_dim=1024,rows=5,cols=4,overlap_ratio=0.12):
    x = np.arange(rows*cols).reshape(rows,cols)
    loc = np.where(x==tile_idx)
    i = loc[0]; j=loc[1]
    
    #initialize crop to false
    left_crop = False
    top_crop = False
    
    # conditions
    if i == 0 and j > 0:
        left_crop = True
    elif j == 0 and i > 0:
        top_crop = True
    elif i>0 and j>0:
        top_crop = True
        left_crop = True
    
    # make a mask
    mask = np.zeros((xy_dim,xy_dim),dtype='bool')  
    if top_crop:
        mask[:round(xy_dim*overlap_ratio),:] = True
    if left_crop:
        mask[:,:round(xy_dim*overlap_ratio)] = True
    
    return mask


def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D





def spectrum_plus(spectrum,radii=[1],sigma=1):
    # x and y dimensions
    xdim = spectrum.shape[0]
    ydim = spectrum.shape[1]
    zdim = spectrum.shape[2]
    
    # determine the largest radius
    maxr = np.max(np.asarray(radii))
    
    # initialize an array to the save the new features
    spectrum_extended = np.zeros((xdim,ydim,zdim*len(radii)))
    
    for idx,r in enumerate(radii):
        spectrum_extended[:,:,zdim*idx:(zdim*idx)+zdim] = GaussFilterCube(spectrum,size=r*2+1,sigma=2)
        
        #for i in range(maxr,spectrum.shape[0]-maxr):
        #    for j in range(maxr,spectrum.shape[1]-maxr):
        #        #spectrum_extended[i,j,zdim*idx:(zdim*idx)+zdim] = np.mean(spectrum[i-r:i,j-r:j,:],axis=(0,1))
        #        #spectrum_extended[i,j,zdim*idx:(zdim*idx)+zdim] = np.mean(spectrum[i-r:i+r,j-r:j+r,:],axis=(0,1))
        #     
        #        w = np.repeat(gaussian_kernel(2*r+1,std=sigma)[:, :, np.newaxis],zdim, axis=2)
        #        spectrum_extended[i,j,zdim*idx:(zdim*idx)+zdim] = np.average(spectrum[i-r:i+r+1,j-r:j+r+1,:],axis=(0,1),weights=w)
    
    return spectrum_extended



def pad_spectrum(spectrum,pad_width=10):
    # x and y dimensions
    xdim = spectrum.shape[0]
    ydim = spectrum.shape[1]
    zdim = spectrum.shape[2]
    
    spectrum_padded = np.zeros((xdim+2*pad_width,ydim+2*pad_width,zdim))
    for k in range(zdim):
        spectrum_padded[:,:,k] = np.pad(spectrum[:,:,k],pad_width=pad_width,mode='mean')
    return spectrum_padded


def SAD(s1, s2):   # find the source later
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle


def spectrum_dog(spectrum_extended,originalzdim = 250):   # not used recently, but maybe useful
    xdim = spectrum_extended.shape[0]
    ydim = spectrum_extended.shape[1]
    zdim = spectrum_extended.shape[2]

    # the number of radii
    radii_num = int(zdim/originalzdim)

    # initialize the dog 
    spectrum_dog = np.zeros((xdim,ydim,zdim-originalzdim))

    for idx in range(radii_num-1):
        idx1 = idx+1
        innerGauss = spectrum_extended[:,:,originalzdim*idx:(originalzdim*idx)+originalzdim]
        outerGauss = spectrum_extended[:,:,originalzdim*idx1:(originalzdim*idx1)+originalzdim]
        spectrum_dog[:,:,originalzdim*idx:(originalzdim*idx)+originalzdim] = outerGauss - innerGauss 

    return spectrum_dog


def pad_spectrum(spectrum,pad_width=10,mode='mean'):
    # x and y dimensions
    xdim = spectrum.shape[0]
    ydim = spectrum.shape[1]
    zdim = spectrum.shape[2]
    
    spectrum_padded = np.zeros((xdim+2*pad_width,ydim+2*pad_width,zdim))
    for k in range(zdim):
        spectrum_padded[:,:,k] = np.pad(spectrum[:,:,k],pad_width=pad_width,mode=mode)
    return spectrum_padded


def spectrum_weighted(spectrum,r=1,sigma=1,padmode='edge'):
    # weighting by spectral similarity to yo neighbors
    # x and y dimensions
    xdim = spectrum.shape[0]
    ydim = spectrum.shape[1]
    zdim = spectrum.shape[2]
    ws = 2*r+1
    
    # the mean spectrum (same terminlolgy as the paper)
    I = np.mean(spectrum,axis=(0,1))
    
    # pad the spectrum by r
    spectrum = pad_spectrum(spectrum,pad_width=r,mode=padmode)
    
    # initialize an array to the save the new features
    spectrum_weighted = np.zeros(spectrum.shape)
    
    # save the rho array (just to look at it)
    rho_img = np.zeros((spectrum.shape[0],spectrum.shape[1]))
    
    # the spatial weight for each pixel (Gaussian for now)
    #beta = gaussian_kernel(2*r+1,std=sigma)
    beta = get_beta(r)
    
    for i in range(r,spectrum.shape[0]-r):
        for j in range(r,spectrum.shape[1]-r): 
            
            # the spectrum in the ws square
            sub_spectrum = spectrum[i-r:i+r+1,j-r:j+r+1,:]
            gamma = np.zeros((ws,ws))
            for ii in range(sub_spectrum.shape[0]):
                for jj in range(sub_spectrum.shape[1]):
                    gamma[ii,jj] = SAD(spectrum[i,j,:],sub_spectrum[ii,jj])
  
            # calculate weight and modify
            alpha = np.sum(np.multiply(beta,gamma))
            rho = (1+math.sqrt(alpha))**2
            rho_img[i,j] = rho

            spectrum_weighted[i,j,:] = (1/rho)*(spectrum[i,j,:]-I)+I
    
    # return the weighted spectrum after unpadding
    return spectrum_weighted[r:-r,r:-r,:],rho_img[r:-r,r:-r]

def get_beta(r=1):
    ws = 2*r+1
    beta = np.ones((ws,ws))
    for i in range(ws):
        for j in range(ws):
            if i != r or j != r:
                beta[i,j] = 1/((i-r)**2+(j-r)**2)
    return beta
    

    
def save_tiff_stack(spectrum,outpath):
    with tif.TiffWriter(outpath) as tf:
      for i in range(spectrum.shape[2]):
        filename = f"image_{i}"
        img = normalize8(spectrum[:,:,i])
        tf.save(img, photometric='minisblack', description=filename, metadata=None)
        
        
def sparsity(spectrum):
    NumelSpectrum = spectrum.shape[0]*spectrum.shape[1]*spectrum.shape[1]
    return 100-(np.count_nonzero(spectrum)/NumelSpectrum)*100



def vca_purity_mask(spectrum_2D,nEM=10,special=False):
    n = spectrum_2D.shape[0]
    end_members_vca,_,_ = vca(spectrum_2D[:,:].transpose(),nEM,verbose = True)
    Ends = np.array([nnls(end_members_vca,i)[0] for i in spectrum_2D])
    best_fit_endmemebr = np.argmax(Ends,axis=1)
    sam_to_best_endmember = np.zeros(best_fit_endmemebr.shape)
    for i in range(n):
        sam_to_best_endmember[i] = SAD(spectrum_2D[i,:],end_members_vca[:,best_fit_endmemebr[i]])
    sam_to_best_endmember = (sam_to_best_endmember - sam_to_best_endmember.min())/(sam_to_best_endmember.max()-sam_to_best_endmember.min())
    if special:
        sam_to_best_endmember = np.abs((sam_to_best_endmember - 0.5))*2
    return sam_to_best_endmember 


#### for frame count analysis

def compute_inner_outer_similarity_with_distances(dist_to_ref, labels):
    # Initialize lists to store inner-class similarities and outer-class dissimilarities
    inner_class_similarities = []
    outer_class_dissimilarities = []
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Compute inner-class similarity and outer-class dissimilarity for each cluster
    for label in unique_labels:
        # Get indices of data points belonging to the current cluster
        cluster_indices = np.where(labels == label)[0]
        
        # Get distances from the reference for data points in the current cluster
        cluster_distances = dist_to_ref[cluster_indices]
        
        # Compute average distance within the cluster
        intra_cluster_distances = np.mean(np.abs(np.subtract.outer(cluster_distances, cluster_distances)))
        inner_class_similarities.append(intra_cluster_distances)
        
        # Get distances from the reference for data points in other clusters
        other_cluster_indices = np.where(labels != label)[0]
        other_cluster_distances = dist_to_ref[other_cluster_indices]
        
        # Compute average distance to points in other clusters
        inter_cluster_distances = np.mean(np.abs(np.subtract.outer(cluster_distances, other_cluster_distances)))
        outer_class_dissimilarities.append(inter_cluster_distances)
    
    # Compute the ratio of inner-class similarity to outer-class dissimilarity
    similarity_dissimilarity_ratio = np.mean(inner_class_similarities) / np.mean(outer_class_dissimilarities)
    
    return similarity_dissimilarity_ratio

# Euclidean distance
def euc(array1, array2):
    return np.sqrt(np.sum((array1 - array2)**2))