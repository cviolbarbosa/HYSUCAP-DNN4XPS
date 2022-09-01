from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import nlopt
import ipywidgets as widgets

DX = 0.01

def load_file_data(CSV_COLUMN_SEPARATOR, ENERGY_AXIS):
    # Load your experimental data
    experimental_data = './input/experimental_data.csv'
    csv_panda = pd.read_csv(experimental_data, sep=CSV_COLUMN_SEPARATOR, header=None)

    # Normalize experimental data
    spectrum_e = csv_panda.iloc[:,0].to_numpy()
    spectrum =   csv_panda.iloc[:,1].to_numpy()
    spectrum = spectrum - np.min(spectrum)
    spectrum = spectrum /np.max(spectrum) 

    if np.sum(np.diff(spectrum_e)) < 0:
        spectrum_e = np.flip(spectrum_e)
        spectrum = np.flip(spectrum)
    
    y_interpolated = np.interp(ENERGY_AXIS,spectrum_e, spectrum)
       
    return y_interpolated, ENERGY_AXIS


# FIT LOSS FUNCTION
def fit_loss(x, y_true, model, energy_axis, atomic_positions):
    y_spectrum = CalculatedSpectrum(model, x, energy_axis, atomic_positions)
    background, difference = y_spectrum.infer_background(y_true)
    deviation = np.square(difference - background)
    back_error = np.sum(deviation)
    mae = np.sum(np.abs(difference))

    loss = 0.6*mae + 0.4*back_error
    
    print(f"mae: {mae} bkg_error: {back_error}")
    return float(loss)



def calculate_spectra(model, x, gaussian=1, offset=0, shift=0, num_atoms=1, input_size=7):
    y_pred = None
    lim = input_size + 1 # X contains NN input plus one parameter for scale factor
  
    for i in range(0, num_atoms):
        weight_i = x[i*lim + input_size]
        x_tensor = np.expand_dims(x[i*lim:i*lim+input_size],0)
        if y_pred is None:
            y_pred = weight_i * model(x_tensor)[0].numpy()[:,0]
        else:
            y_pred += weight_i * model(x_tensor)[0].numpy()[:,0]
                   
    y_pred = shift_spectrum_energy(y_pred,shift)
    y_pred = y_pred + offset
    y_pred = gaussian_broadening(y_pred, gaussian)
    return y_pred   


def gaussian_broadening(y, broad):
    FWHM = broad/DX
    sigma = FWHM/2.355
    return gaussian_filter1d(y, sigma)
    


def shift_spectrum_energy(spec, shift):
    dx = DX
    size = len(spec)
    if abs(shift) < dx:
        return spec    
    pad = np.zeros(int(abs(shift)/dx))
    if shift < 0:
        b = np.concatenate((spec,pad))
        b = b[-size:]
    else:
        b = np.concatenate((pad, spec))
        b = b[0:size]
    return b


def parse_opt_input(x):
    NN_input = x[0:7]
    gs_broad= x[7]
    scale =   x[8]
    offset =  x[9]; 
    shift =   x[10] 
    return NN_input, gs_broad, scale, offset, shift 


def create_optimizer(S,fit_loss, num_atoms=1):
    """ Create nlopt optimizer for spectrum fit """
    
    lower_state =    [S['range_nox'][0],
                      S['range_delta'][0],
                      S['range_Udd'][0],
                      S['range_Upd'][0],
                      S['range_T2q'][0],
                      S['range_Dt'][0],
                      S['range_Ds'][0],
                      S['range_scale'][0],
                      
                      ]
   
    lower_instrum =  [S['range_broad'][0],
                      S['range_offset'][0],
                      S['range_shift'][0]  
                      ]
    
    lower = np.array(num_atoms*lower_state + lower_instrum)

    upper_state =    [S['range_nox'][1],
                      S['range_delta'][1],
                      S['range_Udd'][1],
                      S['range_Upd'][1],
                      S['range_T2q'][1],
                      S['range_Dt'][1],
                      S['range_Ds'][1],
                      S['range_scale'][1],
                     ]
   
    upper_instrum =  [S['range_broad'][1],
                      S['range_offset'][1],
                      S['range_shift'][1]  
                      ]
    
    upper = np.array(num_atoms*upper_state + upper_instrum)    
    
    
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(upper))  
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_ftol_rel(S['set_ftol_rel'])
    opt.set_xtol_rel(S['set_xtol_rel'])
    opt.verbose = 1
    # opt.maxeval = 1
    opt.set_min_objective(fit_loss)
    x0 = (upper + lower) / 2
    return opt, x0


class AtomicElement():
    def __init__(self, nox=2, delta=0, Udd=0, Upd=0, T2q=0,Dt=0, Ds=0, weight=1):
        self.nox = nox
        self.delta = delta
        self.Udd = Udd
        self.Upd = Upd
        self.T2q = T2q
        self.Dt = Dt
        self.Ds = Ds
        self.weight = weight

    
    
class CalculatedSpectrum():
    y_peaks = None
    y_background = None
    spectrum = None
    atoms = []
    
    def __init__(self, model, X, energy_range, atomic_positions=1, input_size=7):
        self.num_atoms = atomic_positions
        self.model = model
        self.energy_range = energy_range
        self.atoms= []        
        lim = input_size + 1 # NN_input parameters and Scale factor
        
        for i in range(0,atomic_positions):
            if len(X):
                self.atoms.append(AtomicElement(*X[i*lim:(i+1)*lim]))  
            else:
                atom = AtomicElement()
                print(atom)
                self.atoms.append(atom)  
        if len(X):
            self.broad = X[-3] 
            self.offset = X[-2] 
            self.shift = X[-1]
        else:
            self.broad = 0.05 
            self.offset = 0 
            self.shift = 0
        
        
    def __str__(self):
        inst_params =  (f"\nINSTRUMENTAL PARAMETERS:\n"
                        f"broad: {self.broad}\n"
                        f"offset: {self.offset}\n"
                        f"shift: {self.shift}\n")
        ele_params = ""
        for i,atom in enumerate(self.atoms):
              ele_params += (f"\nELECTRONIC PARAMETERS Element:{i}\n"
                f"Ox. state:{round(atom.nox,2)}\n" 
                f"delta: {atom.delta}\n"
                f"Udd: {atom.Udd}\n"
                f"Upd: {atom.Upd}\n"
                f"\nCRYSTAL PARAMS:\n"
                f"T2q: {atom.T2q}\n"
                f"Dt: {atom.Dt}\n"
                f"Ds: {atom.Ds}\n"
                f"Weight:{atom.weight}\n" 
                )
                
        return inst_params + ele_params
                
        
    def get_model_input(self):
        nn_input = []
        for atom in self.atoms:
            nn_input += [atom.nox, 
                        atom.delta,
                        atom.Udd,
                        atom.Upd,
                        atom.T2q,
                        atom.Dt,
                        atom.Ds,
                        atom.weight
                       ]
     
        return np.array(nn_input)
    
    def get_parameter_array(self):                
        nn_input = self.get_model_input()
        instrum = np.array([self.broad, self.offset, self.shift]);
        parameters = np.concatenate((nn_input, instrum)) 
        
        return parameters
    
    def calculate_peaks(self):
        model_input = self.get_model_input()
        self.y_peaks = calculate_spectra(self.model, model_input, self.broad, self.offset, self.shift, self.num_atoms)
        return self.y_peaks
        
        
    def infer_background(self, y_experiment):
        self.calculate_peaks()
        difference = (y_experiment - self.y_peaks)
        self.y_background = savgol_filter(difference,301,1)
        self.y_background = savgol_filter(self.y_background,301,1)
        return self.y_background, difference
             
    def calculate_spectra(self, y_experiment=None):
        self.spectrum = self.calculate_peaks()
        if y_experiment is not None:
            self.infer_background(y_experiment)
            self.spectrum = self.spectrum + self.y_background
        return self.spectrum
        
        
    def plot(self, plt,y_experiment=None):
        if self.y_background is None and y_experiment is None:
            plt.plot(self.energy_range, self.spectrum)
        if self.y_background is not None and y_experiment is None:
            plt.plot(self.energy_range, self.spectrum, self.energy_range, self.y_background)
            plt.legend(['Calculated spectrum', 'Background estimation'])
            
        if self.y_background is not None and y_experiment is not None:
            plt.plot(self.energy_range, self.spectrum, self.energy_range, self.y_background, self.energy_range, y_experiment)  
            plt.legend(['Calculated spectrum', 'Background estimation', 'Experimental data'])
        
        plt.xlabel('energy (eV)')
        
        
    
    
uploader = widgets.FileUpload(
    accept='.txt, .dsv, .csv',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False 
)

def on_upload_change(change):
    global uploader
    try:
        input_file = list(uploader.value.values())[0]
    except:
        input_file = uploader.value[0]
    with open("./input/experimental_data.csv", "wb") as fp:
        fp.write(input_file['content'])
        print("File select. Ready to load the file data")

uploader.observe(on_upload_change, names=['value'])



    