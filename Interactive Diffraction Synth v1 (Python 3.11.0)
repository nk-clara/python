###################### Interactive Diffraction Synth ####################

## IMPORTING LIBRARIES
import numpy as np
import numpy.fft as fft

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import sounddevice as sd
sd.default.latency = 'high'

import pygame


## GETTING THE MODULATED SOUND 

def aperture_function(N,a,b,padding):
    """
    Define a one-dimensional aperture centered at x = 0.

    Parameters
        N: number of slits
        a: slit width
        b: slit separation
        padding: determines how far the sample range extends beyond the aperture

    Returns
        x: x-coordinates (in meteres) of the samples taken along the aperture axis
        aperture: the aperture function as a boolean array

    Note:   currently only works for single and double slit apertures
            (with a transmission of either 0 or 1 at a given x-coordinate)

    """

    # Defining a sample range along the aperture axis, centered at x = 0 (meters)
    x_range = padding*a # [m]
    dx = a/100 # [m]
    x = np.arange(-x_range/2,x_range/2,dx)

    # Determining the aperture function
    if N == 1:
        aperture = np.abs(x) < a/2
    if N == 2:
        aperture = (np.abs(x) > (b-a/2)) & (np.abs(x) < (b+a/2))

    return x, aperture


def aperture_fft(x,aperture,l,d,cutoff):
    
    # Calculate magnitude of wavevector [rad/m]
    k = 2 * np.pi / l 

    # Defining discrete spatial frequency bins for the FFT (rad/m)
    kx = 2 * np.pi * fft.fftshift(fft.fftfreq(len(x),np.diff(x)[0]))

    # Complex electric field at screen from FFT (arbitrary units)
    E = fft.fftshift(fft.fft(aperture))

    # Normalized electric field amplitude (arbitrary units)
    ampl = np.abs(E)/np.max(np.abs(E))

    # Converting spatial frequencies into x-positions along the screen (meters)
    X =  (kx * d / k)

    # Selecting a relevant range of the screen to observe
    mask = np.abs(X) < cutoff
    
    return X[mask], ampl[mask]


def modulate_sound(x,aperture,l,d,duration,carrier_freq,scale):
    """
    Amplitude modulation of a pure sine based on a Fraunhofer amplitude pattern

    Parameters
        x: x-coordinates (in meteres) of the samples taken along the aperture axis
        aperture: takes an array containing the aperture function values
        duration: duration in seconds of the audio
        l: wavelength of light used for Fraunhofer amplitude pattern
        d: distance from aperture to observation screen for Fraunhofer amplitude pattern
        carrier_freq: frequency of the carrier signal
        scale: determines how the spatial basis is mapped onto a time basis

    Returns

        time_x: array of discrete times at which the modulated amplitude is calculated
        modulated: amplitude of the modulated sine wave (arbitrary units, from 0 to 1)
        sample_rate: number of amplitude samples per unit time

    """

    # Use the space-to-time scaling to determine the
    # relevant range of the Fraunhofer amplitude pattern
    cutoff = scale * duration # [m]

    # Compute FFT
    X, ampl = aperture_fft(x,aperture,l,d,cutoff)
    
    # Setting time as the x-variable
    time_x = np.linspace(0,duration,len(X))
    
    # Amplitude modulation of a pure sine
    pure_sine = np.sin(2 * np.pi * carrier_freq * time_x)
    modulated = pure_sine * ampl

    # Calculating the sample rate 
    sample_rate = int(len(time_x) / duration)

    return modulated, sample_rate


## SETTING BASE PARAMETERS

freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

a = [20e-6,40e-6,60e-6] # [m]
a_index = 0

l = [4e-6,5.5e-6,7e-6] # [m]
l_index = 0

d = [9,18,27] # [m]
d_index = 0

padding = 1000
scale = 15 # [m/s]
duration = 0.3 # [s]
carrier_freq = freqs[0] # [Hz]


x, aperture = aperture_function(1,a[a_index],None,padding)
aperture_index = 0
# Determines which aperture the modulation is based on
# 1: single slit
# 2: double slit

## INITIATING INTERACTIVE BIT

pygame.init()

# Improving sound quality
pygame.mixer.pre_init(44100,-16,2, 1024)
pygame.mixer.init()

# Formatting
screen = pygame.display.set_mode((600, 700))
pygame.display.set_caption("Diffraction Synthesizer")

WHITE = (255,255,255)
BLACK = (0,0,0)
PANEL1 = (211,211,211)
PANEL2 = (120, 120, 120)
LIGHT = (64, 194, 237)
DARK = (52, 157, 191)

font = pygame.font.SysFont("Bahnschrift", 25)

# Set up controls
def control_panel(x1,y1):
    # background
    panel_bg = pygame.Rect(x1, y1-38, 440, 158)
    pygame.draw.rect(screen,PANEL2,panel_bg,4)

    # slit width controls
    a_button1 = pygame.Rect(x1+5, y1+5, 140, 33)
    a_button2 = pygame.Rect(x1+5, y1+43, 140, 33)
    a_button3 = pygame.Rect(x1+5, y1+81, 140, 33)
    a_rect = pygame.Rect(x1+5, y1-33, 140, 33)

    pygame.draw.rect(screen, DARK if a_index == 0 else LIGHT, a_button1)
    pygame.draw.rect(screen, DARK if a_index == 1 else LIGHT, a_button2)
    pygame.draw.rect(screen, DARK if a_index == 2 else LIGHT, a_button3)
    pygame.draw.rect(screen, DARK if a_index == 2 else LIGHT, a_button3)
    pygame.draw.rect(screen, PANEL2, a_rect)

    a_label1 = font.render(f"{a[0]:.0e} m", True, WHITE)
    a_label2 = font.render(f"{a[1]:.0e} m", True, WHITE)
    a_label3 = font.render(f"{a[2]:.0e} m", True, WHITE)
    a_label = font.render(f"Slit Width", True, WHITE)

    screen.blit(a_label1, (x1+10, y1+10))
    screen.blit(a_label2, (x1+10, y1+48))
    screen.blit(a_label3, (x1+10, y1+86))
    screen.blit(a_label, (x1+10, y1-25))

    # wavelength controls
    l_button1 = pygame.Rect(x1+150, y1+5, 140, 33)
    l_button2 = pygame.Rect(x1+150, y1+43, 140, 33)
    l_button3 = pygame.Rect(x1+150, y1+81, 140, 33)
    l_rect = pygame.Rect(x1+150, y1-33, 140, 33)

    pygame.draw.rect(screen, DARK if l_index == 0 else LIGHT, l_button1)
    pygame.draw.rect(screen, DARK if l_index == 1 else LIGHT, l_button2)
    pygame.draw.rect(screen, DARK if l_index == 2 else LIGHT, l_button3)
    pygame.draw.rect(screen, DARK if l_index == 2 else LIGHT, l_button3)
    pygame.draw.rect(screen, PANEL2, l_rect)

    l_label1 = font.render(f"{l[0]:.1e} m", True, WHITE)
    l_label2 = font.render(f"{l[1]:.1e} m", True, WHITE)
    l_label3 = font.render(f"{l[2]:.1e} m", True, WHITE)
    l_label = font.render(f"Wavelength", True, WHITE)

    screen.blit(l_label1, (x1+155, y1+10))
    screen.blit(l_label2, (x1+155, y1+48))
    screen.blit(l_label3, (x1+155, y1+86))
    screen.blit(l_label, (x1+155, y1-25))

    # screen distance controls
    d_button1 = pygame.Rect(x1+295, y1+5, 140, 33)
    d_button2 = pygame.Rect(x1+295, y1+43, 140, 33)
    d_button3 = pygame.Rect(x1+295, y1+81, 140, 33)
    d_rect = pygame.Rect(x1+295, y1-33, 140, 33)

    pygame.draw.rect(screen, DARK if d_index == 0 else LIGHT, d_button1)
    pygame.draw.rect(screen, DARK if d_index == 1 else LIGHT, d_button2)
    pygame.draw.rect(screen, DARK if l_index == 2 else LIGHT, d_button3)
    pygame.draw.rect(screen, DARK if d_index == 2 else LIGHT, d_button3)
    pygame.draw.rect(screen, PANEL2, d_rect)

    d_label1 = font.render(f"{d[0]:.0f} m", True, WHITE)
    d_label2 = font.render(f"{d[1]:.0f} m", True, WHITE)
    d_label3 = font.render(f"{d[2]:.0f} m", True, WHITE)
    d_label = font.render(f"Screen Distance", True, WHITE)

    screen.blit(d_label1, (x1+300, y1+10))
    screen.blit(d_label2, (x1+300, y1+48))
    screen.blit(d_label3, (x1+300, y1+86))
    screen.blit(d_label, (x1+300, y1-25))

    explain_txt = font.render(f"Press SPACE to change the aperture shape", True, BLACK)
    screen.blit(explain_txt, (x1+35, y1+125))



# Set up keyboard
key_index = 0

def synth_keys(x2,y2,width,height):
    buffer = 4
    key_width = (width - 2*buffer) / 7
    key_spacing = width / 7

    background = pygame.Rect(x2,y2,width,height)
    pygame.draw.rect(screen,PANEL1,background)

    c_key = pygame.Rect(x2,y2,key_width,height)
    d_key = pygame.Rect(x2+key_spacing,y2,key_width,height)
    e_key = pygame.Rect(x2+key_spacing*2,y2,key_width,height)
    f_key = pygame.Rect(x2+key_spacing*3,y2,key_width,height)
    g_key = pygame.Rect(x2+key_spacing*4,y2,key_width,height)
    a_key = pygame.Rect(x2+key_spacing*5,y2,key_width,height)
    b_key = pygame.Rect(x2+key_spacing*6,y2,key_width,height)

    pygame.draw.rect(screen, PANEL2, c_key, 0 if key_index == "c" else 3)
    pygame.draw.rect(screen, PANEL2, d_key, 0 if key_index == "d" else 3)
    pygame.draw.rect(screen, PANEL2, e_key, 0 if key_index == "e" else 3)
    pygame.draw.rect(screen, PANEL2, f_key, 0 if key_index == "f" else 3)
    pygame.draw.rect(screen, PANEL2, g_key, 0 if key_index == "g" else 3)
    pygame.draw.rect(screen, PANEL2, a_key, 0 if key_index == "a" else 3)
    pygame.draw.rect(screen, PANEL2, b_key, 0 if key_index == "b" else 3)

    C_label = font.render("C", True, BLACK)
    D_label = font.render("D", True, BLACK)
    E_label = font.render("E", True, BLACK)
    F_label = font.render("F", True, BLACK)
    G_label = font.render("G", True, BLACK)
    A_label = font.render("A", True, BLACK)
    B_label = font.render("B", True, BLACK)
    
    screen.blit(C_label, (x2+key_width/2.5,y2+height*0.75))
    screen.blit(D_label, (x2+key_width/2.5+key_spacing,y2+height*0.75))
    screen.blit(E_label, (x2+key_width/2.5+key_spacing*2,y2+height*0.75))
    screen.blit(F_label, (x2+key_width/2.5+key_spacing*3,y2+height*0.75))
    screen.blit(G_label, (x2+key_width/2.5+key_spacing*4,y2+height*0.75))
    screen.blit(A_label, (x2+key_width/2.5+key_spacing*5,y2+height*0.75))
    screen.blit(B_label, (x2+key_width/2.5+key_spacing*6,y2+height*0.75))

    
# Setting up graph display
def plot_to_pygame(x, y, width, height):

    # Graph setup function taken straight from ChatGPT to simplify matplotlib mapping onto pygame

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, duration)
    ax.plot(x, y)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid()

    fig.tight_layout()

    # Render Matplotlib figure to an RGB buffer
    fig.canvas.draw()

    image = np.frombuffer(
        fig.canvas.buffer_rgba(),
        dtype=np.uint8
    )

    image = image.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )

    # Convert to Pygame surface
    surface = pygame.image.frombuffer(
        image[:, :, :3].tobytes(),
        fig.canvas.get_width_height(),
        "RGB"
    )

    plt.close(fig)

    return surface

width = 600
height = 300
audio = 0
sr = 0
graph = plot_to_pygame(0,0,width,height)

# Running the Synth

running = True
while running:

    # Setup visuals
    screen.fill(WHITE)

    x1, y1 = 80,350
    control_panel(x1,y1)
    synth_keys(77,500,450,150)

    if graph is not None:
        screen.blit(graph, (0,0))

    mouse = pygame.mouse.get_pos()

    # Response to events
    for event in pygame.event.get():

        # set aperture function
        if aperture_index == 0:
            x, aperture = aperture_function(1,a[a_index],None,padding)
        elif aperture_index == 1:
            x, aperture = aperture_function(2,a[a_index],3*a[a_index],padding)

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:

            # change value of a (slit width)
            if x1 + 5 <= mouse[0] <= x1 + 145 and y1+5 <= mouse[1] <= y1+38:
                a_index = 0
            if x1 + 5 <= mouse[0] <= x1 + 145 and y1+43 <= mouse[1] <= y1+76:
                a_index = 1
            if x1 + 5 <= mouse[0] <= x1 + 145 and y1+81 <= mouse[1] <= y1+114:
                a_index = 2

            # change value of l (wavelength)
            if x1 + 150 <= mouse[0] <= x1 + 290 and y1+5 <= mouse[1] <= y1+38:
                l_index = 0
            if x1 + 150 <= mouse[0] <= x1 + 290 and y1+43 <= mouse[1] <= y1+76:
                l_index = 1
            if x1 + 150 <= mouse[0] <= x1 + 290 and y1+81 <= mouse[1] <= y1+114:
                l_index = 2

            # change value of l (wavelength)
            if x1 + 295 <= mouse[0] <= x1 + 435 and y1+5 <= mouse[1] <= y1+38:
                d_index = 0
            if x1 + 295 <= mouse[0] <= x1 + 435 and y1+43 <= mouse[1] <= y1+76:
                d_index = 1
            if x1 + 295 <= mouse[0] <= x1 + 435 and y1+81 <= mouse[1] <= y1+114:
                d_index = 2

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and aperture_index == 0:
                aperture_index = 1
            elif event.key == pygame.K_SPACE and aperture_index == 1:
                aperture_index = 0
            
            if event.key == pygame.K_c:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[0],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "c"
            if event.key == pygame.K_d:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[1],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "d"
            if event.key == pygame.K_e:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[2],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "e"
            if event.key == pygame.K_f:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[3],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "f"
            if event.key == pygame.K_g:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[4],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "g"
            if event.key == pygame.K_a:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[5],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "a"
            if event.key == pygame.K_b:
                audio, sr = modulate_sound(x,aperture,l[l_index],d[d_index],duration,freqs[6],scale)
                sd.stop()
                sd.play(audio, sr, blocksize = 2048)
                graph = plot_to_pygame(np.linspace(0,duration,len(audio)), audio, width, height)
                key_index = "b"

    pygame.display.update()


pygame.quit()