"""Reading and preparing LLTEM V1.0 trajectories for Well Clear evaluation.

The pipeline is four steps, in this order:

    load_encounter_pair   read one encounter's two state files from the archive
    extract_states        per-aircraft and relative state at each timestep
    align_to_cpa          window +/-60 s about the metadata CPA, resample to 1 Hz
    estimate_derivatives  Savitzky-Golay smoothing and analytic derivatives

`draw_stratified_sample` selects which encounters to run this on.

All positions are in the runway-centred frame the dataset ships (along-runway,
cross-runway, altitude, in feet), with the runway at 42.4699 deg N,
-71.2874 deg W oriented due north.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# --- preprocessing parameters, as reported in the paper ---------------------
CPA_WINDOW = (-60.0, 60.0)   # seconds either side of the metadata CPA time
DT = 1.0                     # uniform resampling step; matches the native 1 Hz
SG_WINDOW = 7                # Savitzky-Golay window, samples (odd)
SG_ORDER = 3                 # Savitzky-Golay polynomial order
MIN_SAMPLES_IN_WINDOW = 10   # below this an encounter is dropped as too short

# Column names for the extended-state CSVs, in the order the dataset's own
# README lists them.
STATE_COLUMNS = [
    'time', 'speed', 'track_angle', 'bank_angle', 'pitch_angle',
    'accel', 'pos_x', 'pos_y', 'alt', 'vel_x', 'vel_y',
    'vel_z', 'heading_rate', 'lat', 'lon',
]

# State variables carried through alignment.
STATE_VARS = [
    'own_x', 'own_y', 'own_z', 'own_vx', 'own_vy', 'own_vz', 'own_psi', 'own_gamma',
    'int_x', 'int_y', 'int_z', 'int_vx', 'int_vy', 'int_vz', 'int_psi', 'int_gamma',
    'rel_x', 'rel_y', 'rel_z', 'rel_vx', 'rel_vy', 'rel_vz',
    'range_h', 'range_3d', 'bearing', 'closure_rate',
]

# Variables that additionally get a smoothed copy and a time derivative.
DERIV_VARS = [
    'own_x', 'own_y', 'own_z', 'own_vx', 'own_vy', 'own_vz',
    'int_x', 'int_y', 'int_z', 'int_vx', 'int_vy', 'int_vz',
    'rel_x', 'rel_y', 'rel_z', 'range_h', 'range_3d',
]


def load_encounter_pair(zf, encounter_id):
    """Read one encounter's ownship and intruder state files from the archive.

    The raw CSVs carry one header line and a trailing comma on every row, so
    each row parses into 16 fields rather than 15. Passing only 15 names makes
    pandas absorb the first field (time) as the DataFrame index and shift every
    remaining name one position left -- which silently turns `pos_x` into
    cross-runway position, `alt` into a velocity, and `time` into speed. The
    trailing '_junk' name together with skiprows=1 keeps each column aligned
    with its true contents.
    """
    subdir_id = ((encounter_id - 1) // 1000) * 1000 + 1
    subdir = f'encounters_{subdir_id:06d}/'
    own_path = subdir + f'ownstates_{encounter_id:06d}.csv'
    int_path = subdir + f'intstates_{encounter_id:06d}.csv'

    with zf.open(own_path) as f:
        df_own = pd.read_csv(f, names=STATE_COLUMNS + ['_junk'], skiprows=1,
                             skip_blank_lines=True)
    with zf.open(int_path) as f:
        df_int = pd.read_csv(f, names=STATE_COLUMNS + ['_junk'], skiprows=1,
                             skip_blank_lines=True)
    return df_own, df_int


def extract_states(own, intr, encounter_id):
    """Per-aircraft and relative state at every timestep of one encounter."""
    own_gamma = np.arctan2(own['vel_z'].values,
                           np.sqrt(own['vel_x'].values ** 2 + own['vel_y'].values ** 2))
    int_gamma = np.arctan2(intr['vel_z'].values,
                           np.sqrt(intr['vel_x'].values ** 2 + intr['vel_y'].values ** 2))

    rel_x = intr['pos_x'].values - own['pos_x'].values
    rel_y = intr['pos_y'].values - own['pos_y'].values
    rel_z = intr['alt'].values - own['alt'].values
    rel_vx = intr['vel_x'].values - own['vel_x'].values
    rel_vy = intr['vel_y'].values - own['vel_y'].values
    rel_vz = intr['vel_z'].values - own['vel_z'].values

    range_h = np.sqrt(rel_x ** 2 + rel_y ** 2)
    range_3d = np.sqrt(rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
    bearing = np.arctan2(rel_y, rel_x)

    with np.errstate(invalid='ignore', divide='ignore'):
        dot = rel_x * rel_vx + rel_y * rel_vy + rel_z * rel_vz
        closure = -dot / np.where(range_3d > 0, range_3d, np.nan)

    return {
        'encounter_id': encounter_id,
        'time': own['time'].values,
        'own_x': own['pos_x'].values, 'own_y': own['pos_y'].values, 'own_z': own['alt'].values,
        'own_vx': own['vel_x'].values, 'own_vy': own['vel_y'].values, 'own_vz': own['vel_z'].values,
        'own_psi': own['track_angle'].values, 'own_gamma': own_gamma,
        'int_x': intr['pos_x'].values, 'int_y': intr['pos_y'].values, 'int_z': intr['alt'].values,
        'int_vx': intr['vel_x'].values, 'int_vy': intr['vel_y'].values, 'int_vz': intr['vel_z'].values,
        'int_psi': intr['track_angle'].values, 'int_gamma': int_gamma,
        'rel_x': rel_x, 'rel_y': rel_y, 'rel_z': rel_z,
        'rel_vx': rel_vx, 'rel_vy': rel_vy, 'rel_vz': rel_vz,
        'range_h': range_h, 'range_3d': range_3d,
        'bearing': bearing, 'closure_rate': closure,
    }


def align_to_cpa(states, tcpa, window=CPA_WINDOW, dt=DT):
    """Window the encounter about its CPA and resample onto a uniform grid.

    `tcpa` is the CPA time from the encounter metadata, not a recomputed one.
    Returns None when fewer than MIN_SAMPLES_IN_WINDOW samples fall inside the
    window, which happens when CPA sits at the very edge of the trajectory.
    """
    t_rel = states['time'] - tcpa
    mask = (t_rel >= window[0]) & (t_rel <= window[1])
    if mask.sum() < MIN_SAMPLES_IN_WINDOW:
        return None

    t_uniform = np.arange(window[0], window[1] + dt, dt)
    aligned = {'time': t_uniform, 'encounter_id': states['encounter_id']}

    for key in STATE_VARS:
        vals = states[key][mask]
        f = interp1d(t_rel[mask], vals, kind='linear', bounds_error=False,
                     fill_value=(vals[0], vals[-1]))
        aligned[key] = f(t_uniform)
    return aligned


def estimate_derivatives(aligned, window=SG_WINDOW, polyorder=SG_ORDER, dt=DT):
    """Smooth each state variable and take its derivative from the same fit.

    The derivative comes from the local polynomial Savitzky-Golay has already
    fitted, so smoothing and differentiation are one operation rather than a
    filter followed by a finite difference.
    """
    for var in DERIV_VARS:
        sig = aligned[var].copy()
        sig[np.isnan(sig)] = np.nanmean(sig)
        aligned[f'{var}_smooth'] = savgol_filter(sig, window, polyorder)
        aligned[f'd_{var}'] = savgol_filter(sig, window, polyorder, deriv=1, delta=dt)
    return aligned


def draw_stratified_sample(info, geometry, n_total=60000, seed=42):
    """Equal-quota geometry-stratified draw from the unfiltered population.

    Sampling from the full population rather than a pre-filtered "close
    encounters" pool is deliberate: it keeps the sample spanning the whole
    severity range instead of only the encounters already known to be tight.

    No model is fitted anywhere in this analysis, so the sample is not split
    into training and validation subsets.
    """
    pool = info.copy()
    pool['geometry'] = geometry

    geom_types = pool['geometry'].value_counts().index.tolist()
    quota = n_total // len(geom_types)

    parts = [pool[pool['geometry'] == g].sample(min(quota, (pool['geometry'] == g).sum()),
                                                random_state=seed)
             for g in geom_types]
    selected = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(selected) < n_total:
        remaining = pool[~pool['id'].isin(selected['id'])]
        top_up = remaining.sample(min(n_total - len(selected), len(remaining)),
                                  random_state=seed)
        selected = pd.concat([selected, top_up]).sample(frac=1, random_state=seed)
        selected = selected.reset_index(drop=True)
    return selected


def peak_turn_rate_deg_s(aligned):
    """Largest intruder turn rate anywhere in the aligned window, in deg/s.

    Taken from the unwrapped track angle so a wrap through +/-pi does not read
    as an enormous turn. Used to stratify predicted-CPA accuracy: a constant-
    velocity extrapolation should degrade as the intruder manoeuvres harder.
    """
    psi = np.unwrap(aligned['int_psi'])
    return float(np.nanmax(np.abs(np.gradient(psi, DT))) * 180.0 / np.pi)


def preprocess_encounters(zf, selected, progress_every=2000):
    """Run the four preprocessing steps over a selected set of encounters.

    Returns (aligned_list, turn_rate_max, dropped_short, failed), where
    `turn_rate_max` is the per-encounter peak intruder turn rate in deg/s,
    `dropped_short` holds encounters whose CPA sat too close to the trajectory
    edge, and `failed` holds (id, message) pairs for unreadable files. With the
    column alignment correct both of the last two are expected to be empty.
    """
    import time

    aligned_list, turn_rate_max, dropped_short, failed = [], [], [], []
    t0 = time.time()

    for i, row in selected.reset_index(drop=True).iterrows():
        if progress_every and i and i % progress_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(selected) - i)
            print(f'  {i:>6,}/{len(selected):,}  elapsed {elapsed:5.0f}s  eta {eta:5.0f}s')

        enc_id = int(row['id'])
        try:
            own, intr = load_encounter_pair(zf, enc_id)
            states = extract_states(own, intr, enc_id)
            aligned = align_to_cpa(states, float(row['tcpa']))
            if aligned is None:
                dropped_short.append(enc_id)
                continue
            turn = peak_turn_rate_deg_s(aligned)
            aligned = estimate_derivatives(aligned)
            aligned['geometry'] = row['geometry']
            aligned_list.append(aligned)
            turn_rate_max.append(turn)
        except Exception as exc:                      
            failed.append((enc_id, str(exc)))

    print(f'  done in {(time.time() - t0) / 60:.1f} min: '
          f'{len(aligned_list):,} retained, {len(dropped_short):,} short-window, '
          f'{len(failed):,} unreadable')
    return aligned_list, np.asarray(turn_rate_max, dtype=float), dropped_short, failed
