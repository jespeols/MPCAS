import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.pyplot as plt
from os import getcwd


def update_orientations(thetas, particles_in_range, noise, time_step):
    N = len(thetas)
    new_thetas = []
    for i_particle in range(N):
        theta_vector = [thetas[j] for j in particles_in_range[i_particle]]
        theta_av = np.arctan(np.average(np.sin(theta_vector))/np.average(np.cos(theta_vector)))
        
        theta_new = theta_av + noise*np.random.uniform(-1/2, 1/2)*time_step

        new_thetas.append(theta_new)    

    return new_thetas


def update_velocities(v, thetas, V):
    N = len(thetas)
    new_velocities = []
    for i_particle in range(N):
        new_velocities.append([V*np.cos(thetas[i_particle]), V*np.sin(thetas[i_particle])])
    return new_velocities


def update_positions(r, v, L, time_step):
    N = len(r)
    r_new = np.zeros(r.shape)
    for i_particle in range(N):
        r_i = r[i_particle]
        v_i = v[i_particle]

        r_i_new = r_i + v_i*time_step
        # apply periodic boundary conditions
        x = r_i_new[0]
        y = r_i_new[1]
        if x > L:
            x -= L
        elif x < 0:
            x += L
        if y > L:
            y -= L
        elif y < 0:
            y += L
        r_i_new = [x, y]

        r_new[i_particle] = r_i_new

    return r_new


def get_extrapolated_thetas(t, time_step, s, thetas_record, h):
    N = len(thetas_record)
    past_thetas = thetas_record[:, t-s-1:t-1]
    thetas_extrap = np.zeros(N)
    for i_particle in range(N):
        T = np.arange(t-s-1, t-1, time_step)
        Y = past_thetas[i_particle,:]
        lin_model = np.poly1d(np.polyfit(T, Y, 1))

        # theta_extrap = past_thetas[i_particle, -1] - h*dt
        theta_extrap = lin_model(t-1-h)

        thetas_extrap[i_particle] = theta_extrap

    return thetas_extrap


def calc_alignment_coeff(v, V):
    N = len(v)
    v_align = [np.divide(v_i, V) for v_i in v] 

    return 1/N * np.linalg.norm(sum(v_align))


def calc_clustering_coeff(r, R_f):
    N = len(r)
    vor_areas = get_Voronoi_areas(r)
    count = sum(map(lambda x: x < np.pi*R_f**2, vor_areas))
    return count/N


def calculate_polygon_area(polygon):
    n = len(polygon) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = np.abs(area) / 2.0
    return area


def get_Voronoi_areas(r):

    vor = Voronoi(r)
    vor_areas = []
    for region in vor.regions:  # i for particle, reg for its corresponding region index
        polygon = [vor.vertices[j] for j in region]
        # calculate area of polygon
        if -1 not in region:
            region_area = calculate_polygon_area(polygon)
        else:
            region_area = 0
        vor_areas.append(region_area)

    return vor_areas


def plot_particle_config(t, r, R_f, S, noise, folder):  # For Ex3.4-3.7

    w_dir = getcwd()
    vor = Voronoi(r)

    plt.figure()
    voronoi_plot_2d(vor)
    plt.xlabel('x')
    plt.ylabel('y')
    if t==0:
        plt.title('Initial config')
        savename = 'initial_config_eta' + str(noise)
        plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
    elif t==S:
        plt.title('Final config')
        savename = 'final_config_eta' + str(noise)
        plt.savefig(w_dir+'/'+folder+'/Rf'+str(R_f)+'/'+savename+'.png', dpi=300)
    else:
        plt.title(r"config at $t$ = " + str(t) + ', $R_f$ = ' + str(R_f))
        savename = 'config_t' + str(t) + '_eta' + str(noise)
        plt.savefig(w_dir+'/'+folder+'/Rf'+str(R_f)+'/'+savename+'.png', dpi=300)
    plt.show()


def plot_particle_config_2(t, r, h, S, folder):   # For Ex3.8 
    w_dir = getcwd()
    vor = Voronoi(r)
    
    plt.figure()
    voronoi_plot_2d(vor)
    plt.xlabel('x')
    plt.ylabel('y')
    if t==0:
        plt.title('Initial config')
        savename = 'initial_config'
        plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
    elif t==S:
        plt.title('Final config, h = ' + str(h))
        savename = 'final_config_h' + str(h)
        plt.savefig(w_dir+'/'+folder+'/h'+str(h)+'/'+savename+'.png', dpi=300)
    else:
        plt.title(r"config at $t$ = " + str(t) + ', h = ' + str(h))
        savename = 'config_t' + str(t) + '_h' + str(h)
        plt.savefig(w_dir+'/'+folder+'/h'+str(h)+'/'+savename+'.png', dpi=300)
    plt.show()


def plot_particle_config_3(t, r, k, S, folder):  # For Ex3.10

    w_dir = getcwd()
    vor = Voronoi(r)

    plt.figure()
    voronoi_plot_2d(vor)
    plt.xlabel('x')
    plt.ylabel('y')
    if t==0:
        plt.title('Initial config')
        savename = 'initial_config'
        plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
    elif t==S:
        plt.title('Final config, k = ' + str(k))
        savename = 'final_config_k' + str(k)
        plt.savefig(w_dir+'/'+folder+'/k'+str(k)+'/'+savename+'.png', dpi=300)
    else:
        plt.title(r"config at $t$ = " + str(t) + ', $k$ = ' + str(k))
        savename = 'config_t' + str(t) + '_k' + str(k)
        plt.savefig(w_dir+'/'+folder+'/k'+str(k)+'/'+savename+'.png', dpi=300)
    plt.show()