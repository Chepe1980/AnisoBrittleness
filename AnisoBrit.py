import streamlit as st
import numpy as np
import lasio
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("Geomechanical Anisotropy Analysis")

# Helper functions
def calculate_poissons_ratio(vp, vs):
    """Calculate Poisson's ratio from Vp and Vs"""
    return (vp**2 - 2*vs**2) / (2*(vp**2 - vs**2))

def calculate_youngs_modulus(vp, vs, rho):
    """Calculate Young's Modulus (E) in GPa"""
    return (rho * vs**2 * (3*vp**2 - 4*vs**2) / (vp**2 - vs**2)) / 1e9

def normalize(x):
    """Min-max normalization to [0,1] range"""
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# File upload
uploaded_file = st.file_uploader("Upload LAS file", type=['las', 'LAS'])
if not uploaded_file:
    st.stop()

# Read LAS file
try:
    las = lasio.read(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
except Exception as e:
    st.error(f"Error reading LAS file: {e}")
    st.stop()

# Check required curves
required_curves = ['DTPMOD', 'DTSMOD', 'RHOBMOD']
missing_curves = [curve for curve in required_curves if curve not in las.keys()]
if missing_curves:
    st.error(f"Missing required curves: {', '.join(missing_curves)}")
    st.stop()

# Extract data
Vp = 304800 / las['DTPMOD']  # Convert μs/ft to m/s
Vs = 304800 / las['DTSMOD']  # Convert μs/ft to m/s
Rho = las['RHOBMOD'] * 1000  # Convert g/cm³ to kg/m³
depth = las.index

# Calculate basic parameters
valid_velocity_mask = (Vp > 0) & (Vs > 0) & (Vp > Vs*np.sqrt(2))
PR = np.full_like(Vp, np.nan)
PR[valid_velocity_mask] = calculate_poissons_ratio(Vp[valid_velocity_mask], Vs[valid_velocity_mask])

YM = np.full_like(Vp, np.nan)
YM[valid_velocity_mask] = calculate_youngs_modulus(Vp[valid_velocity_mask], Vs[valid_velocity_mask], Rho[valid_velocity_mask])

# Calculate Vp/Vs ratio
Vp_Vs = np.divide(Vp, Vs, out=np.zeros_like(Vp), where=(Vs != 0))

# Calculate Thomsen parameters
with np.errstate(divide='ignore', invalid='ignore'):
    delta = ((1 + 3.87 * Vp_Vs - 5.54)**2 - (Vp_Vs**2 - 1)**2) / (2 * Vp_Vs**2 * (Vp_Vs**2 - 1))
    epsilon = 0.2090 * Vp_Vs - 0.2397
    gamma = 0.4014 * Vp_Vs - 0.5576
    g = (Vs**2/Vp**2)
    
    # Calculate anisotropic parameters
    term1 = (delta/(1-2*g))
    term2 = ((epsilon-gamma*g**2)/(1-g**2))
    Vv = PR*(1+term1-term2)
    Vh = PR*(1 + (epsilon/((2*g**2)*(1-2*g**2)*(1-g**2))) - 
             (2*gamma/(1-2*g**2)) - (delta/(1-g**2)))
    
    term6 = (4*PR*delta)
    term7 = ((4*PR**2)*(epsilon-g*gamma))
    YMv = (YM - term6 + term7)
    
    term8 = (-1*epsilon)*((1-2*g)*delta)
    term9 = (2*g*(3-4*g)*(1-g))
    term10 = 4*(1-2*g)*gamma
    term11 = (3-4*g)
    YMh = (YM*(1+(term8/term9)+(term10/term11)))

# Normalize parameters
valid_mask = ~np.isnan(YMv)
YMv_norm = np.full_like(YMv, np.nan)
YMh_norm = np.full_like(YMh, np.nan)
Vv_norm = np.full_like(Vv, np.nan)
Vh_norm = np.full_like(Vh, np.nan)

YMv_norm[valid_mask] = normalize(YMv[valid_mask])
YMh_norm[valid_mask] = normalize(YMh[valid_mask])
Vv_norm[valid_mask] = normalize(Vv[valid_mask])
Vh_norm[valid_mask] = normalize(Vh[valid_mask])

# Calculate brittleness
BRITv = (YMh_norm + Vv_norm)/2
BRITh = (YMh_norm + Vh_norm)/2

# Plotting
st.header("Geomechanical Properties Visualization")

# Create figure with 3 tracks
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
fig.suptitle(f"Geomechanical Analysis - {uploaded_file.name}", y=1.02)

# Track 1: Young's Modulus
ax1.plot(YMv, depth, 'r-', label='YMv (Vertical)')
ax1.plot(YMh, depth, 'b-', label='YMh (Horizontal)')
ax1.plot(YM, depth, 'g-', label='YM (Isotropic)')
ax1.set_xlabel("Young's Modulus (GPa)")
ax1.set_ylabel("Depth")
ax1.grid(True)
ax1.legend()
ax1.invert_yaxis()

# Track 2: Poisson's Ratio
ax2.plot(Vv, depth, 'r-', label='Vv (Vertical)')
ax2.plot(Vh, depth, 'b-', label='Vh (Horizontal)')
ax2.plot(PR, depth, 'g-', label='PR (Isotropic)')
ax2.set_xlabel("Poisson's Ratio")
ax2.grid(True)
ax2.legend()

# Track 3: Brittleness
ax3.plot(BRITv, depth, 'r-', label='BRITv (Vertical)')
ax3.plot(BRITh, depth, 'b-', label='BRITh (Horizontal)')
ax3.set_xlabel("Brittleness Index (Normalized)")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# Results summary
st.header("Results Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Young's Modulus (GPa)")
    st.write(f"Vertical (YMv): {np.nanmean(YMv):.2f} ± {np.nanstd(YMv):.2f}")
    st.write(f"Horizontal (YMh): {np.nanmean(YMh):.2f} ± {np.nanstd(YMh):.2f}")
    st.write(f"Isotropic (YM): {np.nanmean(YM):.2f} ± {np.nanstd(YM):.2f}")

with col2:
    st.subheader("Poisson's Ratio")
    st.write(f"Vertical (Vv): {np.nanmean(Vv):.3f} ± {np.nanstd(Vv):.3f}")
    st.write(f"Horizontal (Vh): {np.nanmean(Vh):.3f} ± {np.nanstd(Vh):.3f}")
    st.write(f"Isotropic (PR): {np.nanmean(PR):.3f} ± {np.nanstd(PR):.3f}")

with col3:
    st.subheader("Thomsen Parameters")
    st.write(f"Delta (δ): {np.nanmean(delta):.3f} ± {np.nanstd(delta):.3f}")
    st.write(f"Epsilon (ε): {np.nanmean(epsilon):.3f} ± {np.nanstd(epsilon):.3f}")
    st.write(f"Gamma (γ): {np.nanmean(gamma):.3f} ± {np.nanstd(gamma):.3f}")

# Download results
#output_las = lasio.LASFile()
#output_las.header = las.header
# Copy the header information properly
#output_las.well = las.well
#output_las.curves = las.curves
#output_las.params = las.params
#output_las.version = las.version
#output_las.wrap = las.wrap
import pandas as pd
try:
    # Create DataFrame with all data
    df = pd.DataFrame({
        'DEPT': las.index,
        'YMv': np.nan_to_num(YMv, nan=-999.25),
        'YMh': np.nan_to_num(YMh, nan=-999.25),
        # Add all other curves similarly...
    })
    
    # Create LAS file from DataFrame
    output_las = lasio.LASFile()
    output_las.set_data(df)
    
    # Add well information
    output_las.well.NULL.value = -999.25
    
    st.download_button(
        label="Download Results",
        data=output_las.write(),
        file_name="results.las",
        mime="text/plain"
    )
except Exception as e:
    st.error(f"Download error: {str(e)}")


























    






    
    # Add new calculated curves
    new_curves = [
        ('YMv', YMv, 'GPa', 'Youngs Modulus Vertical'),
        ('YMh', YMh, 'GPa', 'Youngs Modulus Horizontal'),
        ('Vv', Vv, '', 'Poissons Ratio Vertical'),
        ('Vh', Vh, '', 'Poissons Ratio Horizontal'),
        ('BRITv', BRITv, '', 'Brittleness Vertical'),
        ('BRITh', BRITh, '', 'Brittleness Horizontal'),
        ('DELTA', delta, '', 'Thomsen Delta'),
        ('EPSILON', epsilon, '', 'Thomsen Epsilon'),
        ('GAMMA', gamma, '', 'Thomsen Gamma')
    ]
    
    for mnemonic, data, unit, desc in new_curves:
        output_las.add_curve(mnemonic, data, unit=unit)
        output_las.add_param(mnemonic, desc, '')
    
    # Generate download button
    las_data = output_las.write()
    st.download_button(
        label="Download Results (LAS)",
        data=las_data,
        file_name="geomechanical_analysis.las",
        mime="text/plain"
    )
    
except Exception as e:
    st.error(f"Error preparing download file: {str(e)}")
    st.error("Please check the input data and try again.")





output_las.set_data_from_df(las.df())
output_las['YMv'] = YMv
output_las['YMh'] = YMh
output_las['Vv'] = Vv
output_las['Vh'] = Vh
output_las['BRITv'] = BRITv
output_las['BRITh'] = BRITh
output_las['DELTA'] = delta
output_las['EPSILON'] = epsilon
output_las['GAMMA'] = gamma

st.download_button(
    label="Download Results (LAS)",
    data=output_las.write(),
    file_name="geomechanical_analysis.las",
    mime="text/plain"
)
