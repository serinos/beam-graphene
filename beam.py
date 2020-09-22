"""
Title: Simulation Toolkit for Laser Beams Masked by Semi-ablated Saturable Absorbant Surfaces
Author: Onur Serin

Functions:
-- beam_initialize(res, threshold, Ep, w, over_est)  calculates an np.array matrix with I values
wrt x,y coordinates with desired resolution, calculates only first quadrant
values under y=x, then extends by symmetry to the rest of cartesian plane
over_est flag determines over/underestimation, which can be used for error calcs.
Dimension of the matrix is determined by threshold value.
Equation1: J(x,y) = (2*Ep/(pi*(w**2)))*exp((-2*(x**2) - 2*(y**2))/(w**2))
Created beams are Gaussian, return type is Beam

-- beam_initialize_fast(res, threshold, Ep, w) calculates an np.array matrix with I values
wrt x,y coordinates with desired resolution, fast implementation of beam_initialize using
numpy backend, actual Ep values may be more error prone compared to beam_initialize() for
a given resolution

-- beam_inittilt(res, length, Ep, w, deg, is_x)  Returns a Gaussian Beam tilted at some deg(in radians)
If is_x is True, x-y plane tilts arond y-axis, otherwise tilts around x-axis.
This function is also handy for drawing non-tilted beams with matrices with definitive lengths

-- beam_initfunc(res, length, Ep, w, func)  calculates a np.array matrix with dimensions
(res*length + 1, res*length + 1) for a beam, can be used with arbitrary math functions passed through func parameter.
Note that no sanitization is in place. Use beam_initialize() for fast generation of Gaussian beams.
You may use the variable name "const" in your function instead of (2*Ep/(pi*(w**2))) if you provide some Ep
Note that actual Ep will probably be off of the value provided, no correction factors are provided for the given function
The entry at the center of the matrix stands for approximately (0,0) in x-y, +-1 index shift from center
stands for +-(1/res) shift in x-y.
Return type is Beam

-- mask_initialize(beam, <shape params>, thickness, Js, a0, aS, crop)  outputs mask with
desired shape for a given beam, for now only straight lines and circles are implemented
crop (True/False) yields cropped/uncropped masks
Return type is Mask

-- mask_apply(beam, mask)  Applies the following eqn:
Equation2: Jnew := J - deltaJ where deltaJ := J*(a0 + aS/(1 + J/Js))
Returns the emergent Beam

-- integrate_for_energy(beam)  Adds up J values in a beam matrix, returns calculated Ep

-- multi_integrate_for_energy(beamlist)  Returns a list of tuples in format (index, energy)

-- plot_heat(beam or mask)  Plots heat graph of beam/mask

-- plot_energy(beamlist)  Plots energy graph of beams in a list

-- mask_slide(beam, mask, stepsX, stepsY)  Slides mask on beam, returns a tuple of beams, use uncropped masks
May overload the memory

-- mask_slide_iterator(beam, mask, stepsX, stepsY)  Is a generator, slides mask on beam, yields tuples that have
index and resulting beams one by one, use uncropped masks

-- mask_pc_calc(mask)  Calculates the percentage of non-ablated graphene region over the totality of a given mask

-- brewster_calc(n_env, n_mat)  Calculates the brewster angle, takes in n_env and n_mat, returns in radians

-- loss_calc(E_in, E_out, coeff, percent)  Returns loss by comparing E_in and E_out, multiplies by coeff if given,
which is useful for multiplying by 2, which approximates the experimental result as the beam passes twice through a mask.
As for percent, if overwritten as True, the function returns loss in %.


Units: Think of x resolution unit as resolving 1/x um, enter w in um
       Defaults: Js=0.00000015 uJ/um2, Ep=0.04 uJ, res=1, a0=0.01725, aS=0.00575, eval threshold of beam=10^-10uJ
               : n_env=1, n_mat=1.45

Known issues: --Beams initialized by beam_initfunc() cannot be used with circular masks because of dimension mismatch, fix this.
              This is caused by the option shape='circles' assuming every beam.matrix.shape to be of even numbers, which is never
              satisfied for beam_initfunc() unlike beam_initialize()
"""

from time import time
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import matplotlib.pyplot as plt


class DimensionMismatch(Exception): pass
class InsufficientParameters(Exception): pass

class Beam:
    def __init__(self, res, Ep, w, dim, matrix):
        self.res = res
        self.Ep = Ep
        self.w = w
        self.dim = dim
        self.matrix = matrix

    def copy(self):
        return Beam(self.res, self.Ep, self.w, self.dim, self.matrix)

class Mask:
    def __init__(self, shape, dim, matrix, pad, Js, a0, aS, res):
        self.shape = shape
        self.dim = dim
        self.matrix = matrix
        self.pad = pad
        self.Js = Js
        self.a0 = a0
        self.aS = aS
        self.res = res

    def copy(self):
        return Mask(self.shape, self.dim, self.matrix, self.pad, self.Js, self.a0, self.aS, self.res)


def beam_initialize(res=1, threshold=(10**-10), Ep=0.04, w=0, over_est=True):
    if w==0:
        raise DimensionMismatch
    w2 = w**2
    #  By threshold, calculate how many cells will be traversed, assign to
    #  variable "cut", uses Eqn1 to find x and assign int wrt res. (y=0)
    #  For cut, over_est=True always to ensure comparability with over_est=False matrices
    cut = int(np.ceil(res*np.sqrt(((w2)*np.log(2*Ep/(np.pi*(w2)*threshold)))/2)))
    #  Iterate cut times for first line, then use sin function to truncate
    #  after threshold and fill in zeros, will traverse the first quadrant
    #  below y=x and used for filling above y=x,first quad data is put in a list
    #  Over/underestimation fix is done after first quadrant is calculated
    #  with overestimation
    const = np.float32(2*Ep/(np.pi * w2))  # For later use in value assigning with Eqn1
    y_counter = 0
    first_quad = []
    for i in range(cut):
        x_traversal = []
        fix = int(np.sqrt(cut**2 - y_counter**2))  # Used for zeroing out values below thrsh
        for k in range(y_counter):
            x_traversal.append(first_quad[k][y_counter])
        for j in range(y_counter, fix):
            #  Here comes Eqn1 to be assigned to variable "value"
            value = const*np.exp(-2*((j/res)**2 + (y_counter/res)**2)/w2, dtype="float32")
            x_traversal.append(value)
        for l in range(max(fix, y_counter), cut):
            x_traversal.append(0)

        first_quad.append(x_traversal)
        y_counter += 1

    # Fixing for over_est flag:
    if over_est == False:
        _ = first_quad.pop(0)
        first_quad.append([0 for i in range(cut)])
        for i in range(cut):
            _ = first_quad[i].pop(0)
            first_quad[i].append(0)
    # Now, time for extending to other quadrants
    total_matrix = _quadrant_expander(cut, first_quad)

    return Beam(res, Ep, w, cut*2, np.array(total_matrix))


def beam_initfunc(res=1, length=0, Ep=0.04, w=0, func="0"):
    if w==0 or length==0:
        raise DimensionMismatch
    if func=="0":
        raise InsufficientParameters

    w2 = w**2
    const = np.float32(2*Ep/(np.pi * w2))  # Passed funcs might use it

    datapoint_count = res*length
    half_length = length/2

    init_array = (np.linspace(-half_length,half_length, datapoint_count))
    x = init_array
    y = init_array.copy()
    y.shape = (y.shape[0],1)
    x = np.vstack(tuple(x for i in range(x.shape[0])))
    y = np.hstack(tuple(y for i in range(y.shape[0])))

    resultant = eval(func)
    return Beam(res, Ep, w, resultant.shape[0], resultant)


def mask_initialize(Js=0.00000015, a0=0.01725, aS=0.00575, **kwargs):
    try:
        shape = kwargs.pop("shape")
        beam = kwargs.pop("beam")
    except:
        print("Parameters not sufficient")
        raise InsufficientParameters
    try:
        crop_flag = kwargs.pop("crop")
    except:
        crop_flag = True

    if shape=="lines":
        width = kwargs.pop("width")
        thickness = kwargs.pop("thickness")
        digital_thickness = int(np.ceil(thickness * beam.res))  # Note: Ceiling the thickness
        digital_width = int(np.ceil(width * beam.res))
        square_len = digital_thickness + digital_width
        pad = np.vstack((np.zeros((digital_thickness, square_len)), np.ones((digital_width, square_len))))
        mask = _mask_drawlines(pad=pad, dim=beam.dim, crop=crop_flag)

    elif shape=="circles":
        width = kwargs.pop("width")
        thickness = kwargs.pop("thickness")
        digital_thickness = int(np.ceil(thickness * beam.res))  # Note: Ceiling the thickness
        digital_width = int(np.ceil(width * beam.res))
        repet_len = digital_thickness + digital_width
        # Will not use _mask_drawlines() ofc, pad is constructed for sliding reference, same pad with 'lines' case
        pad = np.vstack((np.zeros((digital_thickness, repet_len)), np.ones((digital_width, repet_len))))
        # Quarter circle will be drawn here, then extended to full
        # Also remember that the beam matrices are always of even numbered x,y shape by construction
        if crop_flag is True:
            half_length = beam.dim//2
        else:
            half_length = (beam.dim//2)+int(np.ceil(repet_len/2))  # Uncropped length is defined here in a way
        # that _config_create can calculate sliding offsets in a similar manner to 'lines' case
        first_quad_mask = []
        half_thickness = digital_thickness//2
        for y in range(half_length):
            tmp = []
            for x in range(half_length):
                r_fixed = np.sqrt(x**2 + y**2) - half_thickness
                if (r_fixed % repet_len) >= digital_width:
                    tmp.append(0)
                else:
                    tmp.append(1)
            first_quad_mask.append(tmp)
        # Now, constructing the total matrix from the first quadrant:
        mask = np.array(_quadrant_expander(half_length, first_quad_mask))

    elif shape=="dots":
        pad = np.array([[1,0],[0,1]])
        mask = _mask_drawlines(pad=pad, dim=beam.dim, crop=crop_flag)

    else:
        return 0

    return Mask(shape, mask.shape[0], mask, pad, Js, a0, aS, beam.res)


def mask_apply(beam: Beam, mask: Mask):
    if beam.dim != mask.dim:
        raise DimensionMismatch
    Js = mask.Js
    a0 = mask.a0
    aS = mask.aS
    # Eqn2 will be used
    changed = beam.matrix*mask.matrix
    unchanged = beam.matrix*((-1)*(mask.matrix - 1))
    new_matrix = unchanged + changed*(1-(a0+(aS/(1+(changed/Js)))))

    return Beam(beam.res, beam.Ep, beam.w, beam.dim, new_matrix)


def mask_slide(beam: Beam, mask: Mask, stepsX=0, stepsY=0):  # Pass cropless masks only
    # Build matrix configurations of masks:
    configs = _config_create(mask.pad.shape, beam.dim, mask.dim, stepsY, stepsX)
    # Multiprocessing setup:
    processes = []
    q = Queue()
    cpu = cpu_count()
    ranger = (len(configs)//cpu)+1
    #Timeout fix for process.join, processes will terminate after sufficient time:
    timeout = time()
    mask_c = mask.copy()
    mask_c.matrix = mask.matrix[0:beam.dim, 0:beam.dim]
    mask_c.dim = beam.dim
    _ = mask_apply(beam=beam, mask=mask_c)
    timeout = time() - timeout + 0.02  #  Tweak this if it is too little/too much
    print(f"Best case: {timeout * ranger} seconds")  #  Not precise at all...
    del(_)
    del(mask_c)
    ###

    for i in range(ranger):
        for j in range(cpu):
            try:
                config = configs.pop()
            except:
                break
            process = Process(target=_mask_apply, args=(q, config, beam, mask))
            processes.append(process)
            process.start()
        for process in processes:
            process.join(timeout)  # Timeouts if thread fails to shut down after sufficient time for calculations

    returnee = []

    for i in range(q.qsize()):
        returnee.append(q.get())
    returnee.sort()
    return returnee


def _mask_apply_iter(config, beam, mask):
    #Note that config[0]: Y axis
    mask_c = mask.copy()
    beam_c = beam.copy()
    mask_c.matrix = mask_c.matrix[config[0]:(beam_c.dim + config[0]), config[1]:(beam_c.dim+config[1])]
    mask_c.dim = beam_c.dim
    beam_c = mask_apply(beam_c, mask_c)
    del(mask_c)
    index = config[0] + config[1]
    beam_tuple = (index, beam_c)
    return beam_tuple


def _mask_apply(q, config, beam, mask):
    #Note that config[0]: Y axis
    mask_c = mask.copy()
    beam_c = beam.copy()
    mask_c.matrix = mask_c.matrix[config[0]:(beam_c.dim + config[0]), config[1]:(beam_c.dim+config[1])]
    mask_c.dim = beam_c.dim
    beam_c = mask_apply(beam_c, mask_c)
    del(mask_c)
    index = config[0] + config[1]
    beam_tuple = (index, beam_c)
    q.put(beam_tuple)


def _mask_drawlines(pad: np.ndarray, dim: int, crop=True):
    # Draws a mask by using the pad repetitively to achieve a square matrix,
    # edges have at least 1 and at most 2 extra pads to ensure proper working of mask_slide()
    # crop=1 returns cropped matrix to match dim, cropped section is centralized with an
    # ablated region at the center, which is good for approximating the min loss case
    if pad.shape[0] > dim:
        raise DimensionMismatch
    legroom = pad.shape[0]//2
    pad = np.vstack(tuple(pad for i in range((dim//pad.shape[0])+2)))
    pad = np.hstack(tuple(pad for i in range((dim//pad.shape[1])+2)))
    midpoint = pad.shape[0]//2
    if crop is True:
        best_midpoint_offset = 0
        score = 0  # Will traverse upward and downward till there is a 1, then multiply the two
                   # movement to get a score, max is best for centralization (think of 2*8 vs 5*5)
        for i in range(-legroom,legroom+1):
            cursor_fixed = midpoint + i
            if pad[cursor_fixed,0]==1:
                continue
            cursor = cursor_fixed
            upper = 1
            while True:
                cursor += 1
                if pad[cursor,0]==0:
                    upper += 1
                else:
                    break

            cursor = cursor_fixed
            lower = 1
            while True:
                cursor -= 1
                if pad[cursor,0]==0:
                    lower += 1
                else:
                    break

            score_check = upper*lower
            if score_check > score:
                score = score_check
                best_midpoint_offset = i

        startpoint = midpoint + best_midpoint_offset - dim//2
        return pad[startpoint:startpoint+dim,0:dim]
    else:
        return pad


def plot_heat(beam: Beam):
    high = beam.dim/(2*beam.res)
    low = -1*high
    extent = [low, high, low, high]
    plt.imshow(beam.matrix, cmap='inferno', extent=extent)
    plt.xlabel('Distance (μm)')
    plt.ylabel('Distance (μm)')
    plt.colorbar(label="Energy Density (μJ*μm^-2)")
    plt.show()


def integrate_for_energy(beam: Beam):
    dA = 1 / (beam.res**2)  # dA for integration by adding up squares
    energy = beam.matrix.sum() * dA
    return np.float32(energy)


def _integrate_for_energy(q, index, beam):
    dA = 1 / (beam.res**2)
    energy = beam.matrix.sum() * dA
    q.put((index, np.float32(energy)))


def multi_integrate_for_energy(beamlist):
    # Timeout fix for process.join, processes will terminate after sufficient time
    timeout = time()
    integrate_for_energy(beam=beamlist[0][1])
    timeout = time() - timeout + 0.02  #  Tweak this if it is too little/too much
    print(f"Best case: {timeout * ((len(beamlist)//cpu_count())+1)} seconds")
    ###
    beamlist = beamlist.copy()
    processes = []
    q = Queue()
    cpu = cpu_count()
    ranger = (len(beamlist)//cpu)+1

    for i in range(ranger):
        for j in range(cpu):
            try:
                config = beamlist.pop()
            except:
                break
            process = Process(target=_integrate_for_energy, args=(q, config[0], config[1]))
            processes.append(process)
            process.start()
        for process in processes:
            process.join(timeout)  # Timeouts if thread fails to shut down after sufficient time for calculations

    returnee = []
    for i in range(q.qsize()):
        returnee.append(q.get())
    returnee.sort()
    return returnee


def plot_energy(energylist):
    plt.plot([i for i in range(len(energylist))],[j[1] for j in energylist])
    plt.show()


def mask_slide_iterator(beam: Beam, mask: Mask, stepsX=0, stepsY=0):  # Pass cropless masks only
    # Build matrix configurations of masks:
    configs = _config_create(mask.pad.shape, beam.dim, mask.dim, stepsY, stepsX)

    for i in configs:
        result = _mask_apply_iter(i, beam, mask)
        yield result


def _config_create(maskpadshape: tuple, beamdim: int, maskdim: int, stepsY: int, stepsX: int):
    pad_Y = maskpadshape[0]
    pad_X = maskpadshape[1]
    if (beamdim > maskdim) | ((stepsX == 0) & (stepsY == 0)):
        raise DimensionMismatch
    try:
        step_sizeY = pad_Y//(stepsY)
        if step_sizeY == 0:
            step_sizeY = 1
    except:
        step_sizeY = 0
    try:
        step_sizeX = pad_X//(stepsX)
        if step_sizeX == 0:
            step_sizeX = 1
    except:
        step_sizeX = 0
    try:
        slope = stepsY//stepsX
    except:
        slope = np.inf

    configs = []  # Config elements are tuples: (Y axis, X axis)
    if step_sizeX == 0:
        configs = [(i,0) for i in range(0, pad_Y, step_sizeY)]
    elif step_sizeY == 0:
        configs = [(0,i) for i in range(0, pad_X, step_sizeX)]
    elif step_sizeY > step_sizeX:
        for i in range(0, pad_X, step_sizeX):
            if i*slope <= pad_Y:
                configs.append((i*slope, i))
            else:
                break
    else:
        for i in range(0, pad_Y, step_sizeY):
            if i//slope <= pad_X:
                configs.append((i, int(i//slope)))
            else:
                break
    return configs


def _quadrant_expander(half_length: int, first_quad: list):
    total_matrix = []
    for i in range(half_length):  # 1st quad used for 2nd quad
        total_line = []
        for j in reversed(range(half_length)):
            total_line.append(first_quad[i][j])
        total_matrix.append(total_line + first_quad[i])
    total_matrix.reverse()  # Fixing rotation problem
    for i in reversed(range(half_length)):  # Upper quads used for lower quads
        total_matrix.append(total_matrix[i])
    return total_matrix


def mask_pc_calc(mask: Mask):
       mask.res = 1  # integrate_for_energy() requires res value
       pc = 100*integrate_for_energy(mask)/(mask.dim**2)  # Adds up 1s in the mask,
       # and divides by total number of entries in its matrix to achieve pc of graphene
       del(mask.res)
       return pc


def brewster_calc(n_env=1, n_mat=1.45):
    return np.arctan(n_mat/n_env)


def beam_inittilt(res=1, length=0, Ep=0.04, w=0, deg=1, is_x=True):
    if w==0:
        raise DimensionMismatch
    constant_Ep = np.cos(deg)*(2*Ep/(np.pi*(w**2)))
    minus_2_over_w2 = -2/(w**2)
    half_length = length/2
    datapoint_count = res*length
    init_array = (np.linspace(-half_length,half_length, datapoint_count))**2
    if is_x is True:
        x2_cos2deg = init_array * ((np.cos(deg))**2)
        y2 = init_array
        y2.shape = (y2.shape[0],1)
        resultant = x2_cos2deg + y2
    else:
        x2 = init_array
        y2_cos2deg = init_array * ((np.cos(deg))**2)
        y2_cos2deg.shape = (y2_cos2deg.shape[0],1)
        resultant = x2 + y2_cos2deg

    resultant = constant_Ep*np.exp(minus_2_over_w2*resultant)
    return Beam(res,Ep,w,resultant.shape[0], resultant)


def beam_initialize_fast(res=1, threshold=(10**-10), Ep=0.04, w=0):
    if w==0:
        raise DimensionMismatch
    w2 = w**2
    #  By threshold, calculate the number of points to be traversed
    constant_Ep = 2*Ep/(np.pi*(w**2))
    minus_2_over_w2 = -2/(w**2)
    datapoint_count = int(np.ceil(res*np.sqrt(((w2)*np.log(2*Ep/(np.pi*(w2)*threshold)))/2))*2)
    half_length = (datapoint_count//2)//res

    init_array = (np.linspace(-half_length,half_length, datapoint_count))**2
    x2 = init_array
    y2 = init_array.copy()
    y2.shape = (y2.shape[0],1)
    resultant = x2 + y2
    resultant = constant_Ep*np.exp(minus_2_over_w2*resultant)

    return Beam(res, Ep, w, resultant.shape[0], resultant)


def loss_calc(E_in, E_out, coeff, percent=False):
    if percent is False:
        return coeff*(E_in-E_out)/E_in
    else:
        return coeff*100*(E_in-E_out)/E_in
