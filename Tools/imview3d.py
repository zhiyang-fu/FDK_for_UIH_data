# %%
import torch
import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt

    # plt.figure()
    # plt.title(title)
    # if colorbar:
    #     plt.imshow(array, vmin=vmin, vmax=vmax)
    #     plt.colorbar()
    # else:
    #     plt.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
    # fignums = plt.get_fignums()
    # print('You may need to close Figure %d window to continue...' % fignums[-1])
    # plt.show()


def show_3D_array\
    (array, index=None, tile_shape=None, scale=None, power=None, \
     suptitle=None, titles=None, title_size=None, \
     zyx=None, xlabel=None, ylabel=None, label=None, \
     cmap=None, show=True):
    '''
    Displays a 3D array as a set of z-slice tiles.
    On successful completion returns 0.
    array     : 3D array
    index     : z-slices index, either Python list or string of the form
              : 'a, b-c, ...', where 'b-c' is decoded as 'b, b+1, ..., c';
              : out-of-range index value causes error (non-zero) return
    tile_shape: tuple (tile_rows, tile_columns);
                if not present, the number of tile rows and columns is
                computed based on the array dimensions
    scale     : tuple (vmin, vmax) for imshow; defaults to the range of
                array values
    power     : if present, numpy.power(abs(array), power) is displayed
                (power < 1 improves visibility of relatively small array values)
    suptitle  : figure title; defaults to None
    titles    : array of tile titles; if not present, each tile title is
                label + tile_number
    zyx       : tuple (z, y, x), where x, y, anad z are the dimensions of array
                corresponding to the spatial dimensions x, y and z; zyx=None is
                interpreted as (0, 1, 2)
    xlabel    : label for x axis
    ylabel    : label for y axis
    label     : tile title prefix
    cmap      : colormap
    show      : flag specifying whether the array must be displayed immediately
    '''
    import math
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except:
        print('matplotlib not found, cannot plot the array')
        return
    import numpy

    current_title_size = mpl.rcParams['axes.titlesize']
    current_label_size = mpl.rcParams['axes.labelsize']
    current_xlabel_size = mpl.rcParams['xtick.labelsize']
    current_ylabel_size = mpl.rcParams['ytick.labelsize']
    mpl.rcParams['axes.titlesize'] = 'small'
    mpl.rcParams['axes.labelsize'] = 'small'
    mpl.rcParams['xtick.labelsize'] = 'small'
    mpl.rcParams['ytick.labelsize'] = 'small'

    if zyx is not None:
        array = numpy.transpose(array, zyx)

    nz = array.shape[0]
    if index is None:
        n = nz
        index = range(n)
    else:
        if type(index) == type(' '):
            try:
                index = str_to_int_list(index)
            except:
                return 1
        n = len(index)
        for k in range(n):
            z = index[k]
            if z < 0 or z >= nz:
                return k + 1
    ny = array.shape[1]
    nx = array.shape[2]
    if tile_shape is None:
        rows = int(round(math.sqrt(n*nx/ny)))
        if rows < 1:
            rows = 1
        if rows > n:
            rows = n
        cols = (n - 1)//rows + 1
        last_row = rows - 1
    else:
        rows, cols = tile_shape
        assert rows*cols >= n, \
            "tile rows x columns must be not less than the number of images"
        last_row = (n - 1)//cols
    if scale is None:
        if power is None:
            vmin = numpy.amin(array)
            vmax = numpy.amax(array)
        else:
            vmin = numpy.power(numpy.amin(abs(array)), power)
            vmax = numpy.power(numpy.amax(abs(array)), power)
    else:
        vmin, vmax = scale
    fig = plt.figure()
    if suptitle is not None:
        if title_size is None:
            fig.suptitle(suptitle)
        else:
            fig.suptitle(suptitle, fontsize=title_size)
    for k in range(n):
        z = index[k] #- 1
        ax = fig.add_subplot(rows, cols, k + 1)
        if titles is None:
            if label is not None and nz > 1:
                ax.set_title(label + (' %d' % z))
        else:
            ax.set_title(titles[k])
        row = k//cols
        col = k - row*cols
        if xlabel is None and ylabel is None or row < last_row or col > 0:
            ax.set_axis_off()
        else:
            ax.set_axis_on()
            if xlabel is not None:
                plt.xlabel(xlabel)
                plt.xticks([0, nx - 1], [0, nx - 1])
            if ylabel is not None:
                plt.ylabel(ylabel)
                plt.yticks([0, ny - 1], [0, ny - 1])
        if power is None:
            imgplot = ax.imshow(array[z,:,:], cmap, vmin=vmin, vmax=vmax)
        else:
            imgplot = ax.imshow(numpy.power(abs(array[z,:,:]), power), cmap, \
                                vmin=vmin, vmax=vmax)
    if show:
        fignums = plt.get_fignums()
        last = fignums[-1]
        if last > 1:
            print("You may need to close Figures' 1 - %d windows to continue..." \
                  % last)
        else:
            print('You may need to close Figure 1 window to continue...')
        plt.show()

    mpl.rcParams['axes.titlesize'] = current_title_size
    mpl.rcParams['axes.labelsize'] = current_label_size
    mpl.rcParams['xtick.labelsize'] = current_xlabel_size
    mpl.rcParams['ytick.labelsize'] = current_ylabel_size

    return 0

class ImageSliceViewer3D:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Arguments:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, volume, range='a', figsize=(12,12), cmap='gray'):
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        if range == 'b':
            self.v = [0.015,0.035]
        elif range == 'a':
            self.v = [np.min(volume), np.max(volume)]
        else:
            self.v = range

        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['dim1','dim2', 'dim3'], value='dim1',
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"dim2":[1,0,2], "dim3":[2,0,1], "dim1": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[0] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice,
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False,
            description='Image Slice:'))

    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[z,:,:], cmap=plt.get_cmap(self.cmap),
            vmin=self.v[0], vmax=self.v[1])
        plt.show()



# Create a 3D array with random numbers
# x = np.random.rand(256,256,96)

# ImageSliceViewer3D(x)
