import numpy as np
import tkinter as tk
import time
from PIL import ImageTk
from PIL import ImageGrab
from PIL import Image as im
import imageio

# Define parameters
N = 16      # lattice size
p = 0.001   # growth probability
f = 0.2     # lightning strike probability

window_res = 400

def create_animation_window():
    Window = tk.Tk()
    Window.title("0 fires")
    Window.geometry(str(window_res)+'x'+str(window_res))
    
    return Window


def create_animation_canvas(Window):
    canvas = tk.Canvas(Window)
    canvas.configure(bg="black")
    canvas.pack(expand=True, fill="both")
    
    return canvas
    
        
# Initialize animation window & canvas
animation_window = create_animation_window()
animation_canvas = create_animation_canvas(animation_window)

# create forest
forest_image = np.zeros((N,N,3))
S = np.zeros((N,N))     # State of cells: 0=no tree, 1=tree, 2=burned, 3=burning

# for creating animation gif
iteration = 0
filenames = []

fire_counter = 0
running_simulation = True
while running_simulation:
    # Apply tree growth
    S[(np.random.rand(N,N) < p) & (S == 0)] = 1
        
    lightning_strikes = np.random.rand() < f    # check if lightning strikes
    if lightning_strikes:
        strike_position = np.random.randint(0,N,2)
        tree_struck = S[strike_position[0], strike_position[1]] == 1
        
        if tree_struck:
            fire_counter += 1
            animation_window.title(str(fire_counter)+' fires')
            S[strike_position[0], strike_position[1]] = 3      # The struck tree is burning
            while sum(sum(S==3)) > 0:    # while the fire is spreading
                burning_trees = np.where(S==3)
                for i,j in zip(burning_trees[0].tolist(), burning_trees[1].tolist()):   # loop over burning trees
                    # check upward expansion
                    if i == 0 and S[N-1,j] == 1:   # periodic boundary conditions
                        S[N-1,j] = 3
                    elif S[i-1,j] == 1:
                        S[i-1,j] = 3
                    # check downward expansion
                    if i == N-1 and S[0,j] == 1:
                        S[0,j] = 3
                    elif (i+1) < N-1 and S[i+1,j] == 1:
                        S[i+1,j] = 3
                    # check rightward expansion
                    if j == N-1 and S[i,0] == 1:
                        S[i,0] = 3
                    elif (j+1) < N-1 and S[i,j+1] == 1:
                        S[i,j+1] = 3
                    # check leftward expansion
                    if j == 0 and S[i,N-1] == 1:
                        S[i,N-1] = 3
                    elif S[i,j-1] == 1:
                        S[i,j-1] = 3
                    
                    S[i,j] = 2      # tree is burned
                
    # Render forest image from states
    ### Black background ###
    # forest_image[:,:,:] = 0                 # black background
    # forest_image[:,:,0] = (S == 2)*255      # burned trees = red
    # forest_image[:,:,1] = (S == 1)*255      # trees = green
    ### White background ###
    forest_image[:,:,:] = 255      # white background
    forest_image[S==2,:] = 0
    forest_image[S==2,0] = 255     # Burned trees = red
    forest_image[S==1,:] = 0
    forest_image[S==1,1] = 255     # Trees = green
    
    img = im.fromarray(np.uint8(forest_image),'RGB').resize((window_res,window_res))
    tk_img = ImageTk.PhotoImage(img, master=animation_window)
    animation_canvas.create_image(0, 0, anchor='nw', image=tk_img)
    animation_window.update()
    
    savename = "im_{0:0>6}".format(iteration)
    filenames.append(savename + '.jpg')
    x0 = animation_canvas.winfo_rootx()*1.25
    y0 = animation_canvas.winfo_rooty()*1.02
    x1 = x0 + animation_window.winfo_width()*1.25
    y1 = y0 + animation_window.winfo_height()*1.335

    grab = ImageGrab.grab((x0, y0, x1, y1))
    grab.save(savename + '.jpg')
    
    if sum(sum(S==2)) > 0:
        time.sleep(0.5)
    
    S[S==2] = 0     # Burned trees become empty cells
    
    iteration += 1
    
animation_window.mainloop()

# %% Make GIF/mp4 & remove images

import os

w_dir = os.getcwd()

images = []
sliced_filenames = filenames[3:-1]     # Remove first two and last image
for filename in sliced_filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(w_dir+'\Animations_b)\Animation_p'+str(p)+'_f'+str(f)+'.mp4', images)
imageio.mimsave(w_dir+'\Animations_b)\Animation_p'+str(p)+'_f'+str(f)+'.gif', images)


for file in filenames[0:-1]:
    os.remove(file)