"""
Title: Testing Toolkit for Simulation of Laser Beams Masked by Semi-ablated Saturable Absorbant Surfaces
Author: Onur Serin

Functions:
-- variation_checker(beam, width_config, plot)
Calculates % variation in energy as given masks are translated
in front of a beam for each 1um step one by one, and calculates the % change in loss over mean loss, width_config is a list
with entries (graphene_width, ablation_width). If plot=True, then plots two figures, one for each calculation.

-- simulated_translation_Evals(beam, width_config)
Returns a list of E values after the beam is translated 1um at a time in front of a graphene layer, width_config is a tuple
in the form of (graphene_width, ablation_width). The graphene layer has zebra patterns

-- hypothesis_calculator(hc_a_0,hc_a_s,hc_J_s, hc_E_p, hc_w, hc_res, hc_d_tot, hc_p_gph, hc_c)
Defaults to properties of graphene, the graphene layer has zebra patterns.
Returns simulated data, hypothetical data and error in E_p caused by discretization and finiteness of matrices
The data are E values of the specified beam after passing through a specified mask at positions sequencially 1um apart
Mind that the two data sets will probably not align as hypothetical calculations always start at a predetermined position but
the simulation starts at a semi-random location which depends on the dimension of the mask applied, which is no problem as
we can easily deduce where the maxima of E values must occur.

-- hypothesis_calculator_elliptical(hc_a_0,hc_a_s,hc_J_s, hc_E_p, hc_w, hc_res, hc_d_tot, hc_p_gph, hc_c, hc_theta=0, transpose_mask)
Defaults to properties of graphene, the graphene layer has zebra patterns.
Returns simulated data, hypothetical data and error in E_p caused by discretization and finiteness of matrices
The data are E values of the specified beam after passing through a specified mask at positions sequencially 1um apart
Same alignment issue of hypothesis_calculator() is observed in this function as well.
As for transpose_mask, set it to True to make the the zebra patterns perpendicular to the major axis of the beam, defaults to False
which keeps the major axis parallel

-- generic_tester_Eenc(E_p_vals, w_vals, d_tot_vals, p_gph_vals)
Takes in lists of parameters to traverse each combination of them to calculate %errors at maxima of hypothetical data wrt simulated data
Returns a database pd.DataFrame(columns=['E_p','w','d_tot','p_gph','err%_at_max','err%_at_min'])
See the function definition to take a look at the defaults

-- generic_tester_tilted_Eenc(E_p_vals, w_vals, d_tot_vals, p_gph_vals, theta_vals, transpose_mask)
Takes in lists of parameters to traverse each combination of them to calculate %errors at maxima of hypothetical data wrt simulated data
Returns a database pd.DataFrame(columns=['E_p','w','d_tot', 'theta', 'p_gph','err%_at_max','err%_at_min','err%_Ep'])
See the function definition to take a look at the defaults


Note: Multithread your tests if you can afford the memory space, they use one core only.
Note: Do not import beam.py separately, it is already included here.
"""

from beam import *
from sympy import *
from scipy import integrate
import pandas as pd


# Sympyfying relevant equations:
x,y = symbols('x,y')  # Axes of the graphene plane
E_p,a_0,a_s,J_s,E_tot,p_gph,p_abl,w,theta,d_tot = symbols('E_p,a_0,a_s,J_s,E_tot,p_gph,p_abl,w,theta,d_tot', positive=True)
J = S(2)*E_p*exp((-2/(w**2))*(x**2 + y**2))/(pi*(w**2))  # Energy density at (x,y) for Gaussian beam
J_red = J*(a_0+(a_s)/(S(1) + J/J_s))  # Loss in J after one passage
J_elp = S(2)*E_p*cos(theta)*exp((-2/(w**2))*((x**2)*cos(theta) + y**2))/(pi*(w**2))  # Energy density at (x,y) for Gaussian beam tilted along x axis by theta
J_red_elp = J_elp*(a_0+(a_s)/(S(1) + J_elp/J_s))  # Loss in J after one passage, uses J_elp instead of the function J

# All encompassing hypothesis, explaining translative effects as well:
c = symbols('c')  # Magic constant, low error rates ensue around c=0.006
y_pos = symbols('y_pos') # Relative position of graphene wrt the beam center, center of an ablated region corresponds to d_tot*3/4
E_enc = E_tot*(S(1)+(c*(d_tot-w)/w)*(S(1)-p_gph)*sin(S(2)*pi*y_pos/d_tot))
E_enc_min = E_tot*(S(1)-(c*(d_tot-w)/w)*(S(1)-p_gph))
E_enc_max = E_tot*(S(1)+(c*(d_tot-w)/w)*(S(1)-p_gph))


def variation_checker(beam, width_config, plot=False):  # width_config is a list with entries (graphene_width, ablation_width)
    results = []  # results list will contain energy integral values(wrt offset 1um per step) of each config
    Einput_actual = integrate_for_energy(beam)

    for i in range(len(width_config)):
        mm = mask_initialize(beam=beam, shape='lines', width=width_config[i][0], thickness=width_config[i][1], crop=False)
        stepsize = int(np.ceil(width_config[i][0] + width_config[i][1]))
        itrtr = mask_slide_iterator(beam=beam,mask=mm,stepsY=stepsize)  # Slider slides graphene layer about 1um at a time
        tmp = []
        for j in itrtr:
            k = integrate_for_energy(j[1])
            tmp.append(k)
            del(j)
        results.append(tmp)
        del(mm)
        del(itrtr)

    dE_E = []  # dE_E will hold loss% values calculated from the results list
    for i in range(len(width_config)):
        dE_E.append([(100*(Einput_actual - j)/Einput_actual) for j in results[i]])

    if plot == True:
        for i in range(len(width_config)):
            plt.plot(dE_E[i], label=f"{(width_config[i][0],width_config[i][1])} um")
        plt.ylabel('Loss (%)')
        plt.xlabel('Sliding Offset (um)')
        plt.legend(loc = 'best')
        plt.show()

    dE_E_diff = [100*(max(i)-min(i))/((sum(i))/len(i)) for i in dE_E]
    if plot == True:
        plt.plot(dE_E_diff,'o')
        plt.ylabel(r'% Change in Loss over Mean Loss')
        plt.xlabel('Configuration No')
        plt.show()

    return dE_E, dE_E_diff


def simulated_translation_Evals(beam, width_config):  # width_config is the tuple (graphene_width, ablation_width)
    results = []  # results list will contain energy integral values(wrt offset 1um per step) of each config
    Einput_actual = integrate_for_energy(beam)

    mm = mask_initialize(beam=beam, shape='lines', width=width_config[0], thickness=width_config[1], crop=False)
    stepsize = int(np.ceil(width_config[0] + width_config[1]))
    itrtr = mask_slide_iterator(beam=beam,mask=mm,stepsY=stepsize)  # Slider slides graphene layer about 1um at a time
    for j in itrtr:
        k = integrate_for_energy(j[1])
        results.append(k)
        del(j)

    return results


# Remember that E_enc works for d_tot>w, if w<d_tot then there is only negligible variation with translation
def hypothesis_calculator(hc_a_0=0.01725,hc_a_s=0.00575,hc_J_s=0.00000015, hc_E_p=0, hc_w=0, hc_res=30, hc_d_tot=0, hc_p_gph=0, hc_c=0.006):
    # Defaults to properties of graphene for a_0, a_s, J_s. Defaults the magic constant to c=0.006, and res=30 parts per 1 um
    # Mind that E_enc will err considerably if p_gph<25%, also there is nonlinearity around d_tot \approx w

    # Initialize a beam with beam_initialize_fast, threshold=10**-16
    beam_init = beam_initialize_fast(Ep=hc_E_p, w=hc_w, res=hc_res, threshold=10**-16)
    beam_init_actualEp = integrate_for_energy(beam_init)
    beam_init_Ep_error = 100*np.abs(hc_E_p-beam_init_actualEp)/hc_E_p

    # Numpyfy J_red accordingly
    np_J_red = lambdify((x,y), J_red.subs({a_0:hc_a_0, a_s:hc_a_s, J_s:hc_J_s, w:hc_w, E_p:hc_E_p}), 'numpy')

    # hc_E_tot_ministripes is to be defined next; using the output of integrate_for_energy() for E_p
    hc_E_tot_ministripes = beam_init_actualEp - np.multiply( hc_p_gph, integrate.nquad(np_J_red,[[-np.inf,np.inf],[-np.inf,np.inf]])[0])

    # Return 1um-per-step translation simulation data, hypothetical calculations for these steps, % error of sum of energy density*unit area wrt given E_p
    simulated_translation_data = simulated_translation_Evals(beam_init, ( hc_d_tot*hc_p_gph, hc_d_tot*(1-hc_p_gph) ))
    E_enc_fixed = E_enc.subs({E_tot:hc_E_tot_ministripes, c:hc_c, d_tot:hc_d_tot, w:hc_w, p_gph:hc_p_gph})
    np_E_enc_fixed = lambdify(y_pos, E_enc_fixed, 'numpy')
    hypothetical_translation_data = [np_E_enc_fixed(i) for i in range(int(np.ceil(hc_d_tot)))]

    return simulated_translation_data, hypothetical_translation_data, beam_init_Ep_error


def hypothesis_calculator_elliptical(hc_a_0=0.01725,hc_a_s=0.00575,hc_J_s=0.00000015,\
                                     hc_E_p=0, hc_w=0, hc_res=30, hc_d_tot=0, hc_p_gph=0, hc_c=0.006, hc_theta=0, transpose_mask=False):
    # Defaults to properties of graphene for a_0, a_s, J_s. Defaults the magic constant to c=0.006, and res=30 parts per 1 um
    # Mind that E_enc will err considerably if p_gph<25%, also there is nonlinearity around d_tot \approx w

    # Initialize a beam with beam_inittilt(), length=w*4 for good measure, check if beam_init_Ep_error is good enough
    if transpose_mask == True:
        beam_init = beam_inittilt(Ep=hc_E_p, w=hc_w, res=hc_res, is_x=False, deg=hc_theta, length=int(np.ceil(hc_w*4)))
    else:
        beam_init = beam_inittilt(Ep=hc_E_p, w=hc_w, res=hc_res, deg=hc_theta, length=int(np.ceil(hc_w*4)))
    beam_init_actualEp = integrate_for_energy(beam_init)
    beam_init_Ep_error = 100*np.abs(hc_E_p-beam_init_actualEp)/hc_E_p

    # Numpyfy J_red_elp accordingly
    np_J_red = lambdify((x,y), J_red_elp.subs({a_0:hc_a_0, a_s:hc_a_s, J_s:hc_J_s, w:hc_w, E_p:hc_E_p, theta:hc_theta}), 'numpy')

    # hc_E_tot_ministripes is to be defined next; using the output of integrate_for_energy() for E_p
    hc_E_tot_ministripes = beam_init_actualEp - np.multiply( hc_p_gph, integrate.nquad(np_J_red,[[-np.inf,np.inf],[-np.inf,np.inf]])[0])

    # Return 1um-per-step translation simulation data, hypothetical calculations for these steps, % error of sum of energy density*unit area wrt given E_p
    simulated_translation_data = simulated_translation_Evals(beam_init, ( hc_d_tot*hc_p_gph, hc_d_tot*(1-hc_p_gph) ))
    E_enc_fixed = E_enc.subs({E_tot:hc_E_tot_ministripes, c:hc_c, d_tot:hc_d_tot, w:hc_w, p_gph:hc_p_gph})
    np_E_enc_fixed = lambdify(y_pos, E_enc_fixed, 'numpy')
    hypothetical_translation_data = [np_E_enc_fixed(i) for i in range(int(np.ceil(hc_d_tot)))]

    return simulated_translation_data, hypothetical_translation_data, beam_init_Ep_error


def generic_tester_Eenc(E_p_vals, w_vals, d_tot_vals, p_gph_vals):
    beam_pd = pd.DataFrame(columns=['E_p','w','d_tot','p_gph','err%_at_max','err%_at_min'])
    for test_E_p in E_p_vals:
        for test_w in w_vals:
            for test_d_tot in d_tot_vals:
                for test_p_gph in p_gph_vals:
                    sim_dat, hypo_dat, err_Ep = hypothesis_calculator(hc_E_p=test_E_p,\
                                                                      hc_w=test_w,\
                                                                      hc_d_tot=test_d_tot,\
                                                                      hc_p_gph=test_p_gph,\
                                                                      hc_c=0.006)
                    max_dat = 100*(max(sim_dat)-max(hypo_dat))/max(sim_dat)
                    min_dat = 100*(min(sim_dat)-min(hypo_dat))/min(sim_dat)
                    beam_pd = beam_pd.append({'E_p':test_E_p,\
                                              'w':test_w,\
                                              'd_tot':test_d_tot,\
                                              'p_gph':test_p_gph,\
                                              'err%_at_max':max_dat,\
                                              'err%_at_min':min_dat}, ignore_index=True)
    return beam_pd


def generic_tester_tilted_Eenc(E_p_vals, w_vals, d_tot_vals, p_gph_vals, theta_vals, transpose_mask=False):
    beam_pd_tilted = pd.DataFrame(columns=['E_p','w','d_tot', 'theta', 'p_gph','err%_at_max','err%_at_min','err%_Ep'])
    for test_E_p in E_p_vals:
        for test_w in w_vals:
            for test_d_tot in d_tot_vals:
                for test_p_gph in p_gph_vals:
                    for test_theta in theta_vals:
                        sim_dat, hypo_dat, err_Ep = hypothesis_calculator_elliptical(hc_E_p=test_E_p,\
                                                                                     hc_w=test_w,\
                                                                                     hc_d_tot=test_d_tot,\
                                                                                     hc_p_gph=test_p_gph,\
                                                                                     hc_theta=test_theta,\
                                                                                     transpose_mask=transpose_mask,\
                                                                                     hc_c=0.006)
                        max_dat = 100*(max(sim_dat)-max(hypo_dat))/max(sim_dat)
                        min_dat = 100*(min(sim_dat)-min(hypo_dat))/min(sim_dat)
                        beam_pd_tilted = beam_pd_tilted.append({'E_p':test_E_p,\
                                                                'w':test_w,\
                                                                'd_tot':test_d_tot,\
                                                                'theta':test_theta,\
                                                                'p_gph':test_p_gph,\
                                                                'err%_at_max':max_dat,\
                                                                'err%_at_min':min_dat,\
                                                                'err%_Ep':err_Ep}, ignore_index=True)  #Use err_Ep to check if dims are good enough
    return beam_pd_tilted
