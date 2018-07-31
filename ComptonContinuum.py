import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

from scipy.ndimage.filters import gaussian_filter

''' A function to calculate the ideal Compton continuum. ''' 
def Klein_Nishina(E_electron,E_ingoing):
    r0 = 2.82e-15 #m
    mc2 = 511
    
    E_outgoing = E_ingoing - E_electron

    if(any(E_ingoing<E_outgoing)):
        print("Error : Incoming gamma-ray energy should be higher than outgoing")
        return
    
    E_ratio = (E_outgoing/E_ingoing)
    
    # Find scattering angle given the energies
    cos_theta = 1 - mc2*(E_ingoing - E_outgoing)/(E_ingoing*E_outgoing)
    # Remove unphysical values
    cos_theta[np.argwhere(np.abs(cos_theta)>1.0)] = float('NaN') 
    # Find the sine of the scattering angle too 
    sin_theta = np.sqrt(1 - cos_theta*cos_theta)
    
    # Calculate the Klein-Nishina Compton continuum 
    g_theta = 0.5 * (E_ratio**2) * (E_ratio + (1.0/E_ratio) - sin_theta*sin_theta)
    prefactor = 2*np.pi*r0*r0*g_theta*sin_theta
    numerator = mc2*((1 + (E_ingoing/mc2)*(1 - cos_theta))**2)
    denominator = E_ingoing*E_ingoing*sin_theta
   
    compton_continuum = prefactor*numerator/denominator

    # Remove non-physical NaNs and set equal to 0 (needed for blurring with resolution )
    compton_continuum[np.argwhere(np.isnan(compton_continuum))] = 0
    
    return compton_continuum


''' A function to blur the ideal Compton continuum. ''' 
def Experimental_Compton_Continuum(E_electron,E_gammaray, res=0.03):
    
    continuum = Klein_Nishina(E_electron,E_gammaray)
#     continuum = np.ones_like(continuum)
    continuum[np.argmin((E_electron - E_gammaray)**2)] = 50*np.max(continuum)
    
    # NaI Resolution is approximately constant from 2 to 3% in 100keV+ range 
    # n.b. this is an approximation and the blurring width is energy dependent
    resolution  = res*E_gammaray
    return gaussian_filter(continuum,sigma=resolution)


    




# The gamma-ray energy from the isotope under consideration     
E_gammaray = 667

# The maximum Compton scattered electron energy 
E_max_Compton = E_gammaray -  (E_gammaray / (1.0 + 2*(E_gammaray/511)))

# The detector energy scale
E_electron = np.linspace(0,1.2*E_gammaray,500)

# Now we plot the results and make it look nice

# Create a 1200x800 figure 
plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')    

# This gets the current axes 
ax = plt.gca()

# Now find the blurred experimental spectrum 
simulated_spectrum = 1e31*Experimental_Compton_Continuum(E_electron,E_gammaray)

# And plot it 
plt.plot(E_electron,simulated_spectrum,'k-')

# Put a grid on the background 
plt.grid()

# This colours in the continuum and photopeak separately 

# These perform the colouring 
ax.fill_between(    E_electron, 
                    simulated_spectrum, 
                    0, where=(E_electron>E_max_Compton),
                    facecolor='indigo', interpolate=True)

max_compton_continuum = E_gammaray
ax.fill_between(    E_electron, 
                    simulated_spectrum, 
                    0, where=(E_electron<E_max_Compton),
                    facecolor='teal', interpolate=True)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts (Arb.)')

# Plot the ideal spectrum too... 

# This calculates the ideal continuum shape 
continuum = Klein_Nishina(E_electron,E_gammaray)

# Add an ideal photopeak 
continuum[np.argmin((E_electron - E_gammaray)**2)] = 1*np.max(continuum)

# This plots a version which is scaled to be the right height to compare to the blurred spectrum 
plt.plot(E_electron,(continuum/sum(continuum))*sum(simulated_spectrum)*0.5,'k:',linewidth=6)

# Add limits and show
plt.xlim([0,max(E_electron)])
plt.ylim([0,1.2])
plt.show()
