#!/anaconda3/envs/oldcoffeaenv/bin/python
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, DelphesSchema#, NanoAODSchema
import pandas as pd
import numpy as np
import os

"""
Step1: read delphes file
"""

def readDelphes(name):
    events = NanoEventsFactory.from_root(
        name,
        schemaclass=DelphesSchema,
        treepath="Delphes",
    ).events()
    return events

def saveAsNPY(result, filename):
    """
    Save the result as a numpy file.
    """
    np.save(filename, ak.to_numpy(result))
    print(f"Events saved to {filename}")

"""
Step2: define functions for event selection
"""

def countItem(arr, item):
    count = 0
    for i in arr:
        if i == item:
            count += 1
    return count
import awkward as ak

def mergeAndMasks(*masks):
    """Merge multiple awkward boolean masks with logical AND."""
    if len(masks) == 0:
        raise ValueError("At least one mask is required")
    
    # 如果传入的是一个列表或元组，解包它
    if len(masks) == 1 and isinstance(masks[0], (list, tuple)):
        masks = masks[0]
    
    merged_mask = masks[0]
    for mask in masks[1:]:
        merged_mask = merged_mask & mask
    return merged_mask

def mergeOrMasks(*masks):
    """OR-combine boolean masks."""
    if len(masks) == 0:
        raise ValueError("At least one mask is required")
    
    # 如果传入的是一个列表或元组，解包它
    if len(masks) == 1 and isinstance(masks[0], (list, tuple)):
        masks = masks[0]
    
    merged_mask = masks[0]
    for mask in masks[1:]:
        merged_mask = merged_mask | mask  # Use | instead of 'or' for element-wise operation
    return merged_mask


def mergeXorMasks(mask1, mask2):
    """
    Merge two boolean masks using logical XOR operation.
    Result is True when exactly one of the masks is True.
    
    Args:
        mask1: First boolean awkward array
        mask2: Second boolean awkward array
        
    Returns:
        awkward array: Combined mask using XOR operation
    """
    merged_mask = mask1 ^ mask2
    return merged_mask

# Additional utility functions for mask operations
def mergeNotMask(mask):
    """Invert a boolean mask."""
    return ~mask

def countPassingEvents(mask):
    """Count True entries in a mask."""
    return ak.sum(mask==True)

def getEfficiency(mask, total_events=None):
    """Compute mask efficiency (passing/total)."""
    passing = ak.sum(mask==True)
    total = total_events if total_events is not None else len(mask)
    return float(passing) / float(total)

def combineMasks(masks_dict, operation='and'):
    """Combine named masks with AND/OR."""
    mask_list = list(masks_dict.values())
    op = operation.lower()
    if op == 'and':
        return mergeAndMasks(*mask_list)
    elif op == 'or':
        return mergeOrMasks(*mask_list)
    else:
        raise ValueError("Operation must be 'and' or 'or'")

print("[INIT] Enhanced mask functions loaded successfully!")

"""
Step3: define functions for event selection
"""
def goodLepton(events, pt_cut=10.0, eta_cut=2.5):
    """
    安全的轻子选择，只使用基本动力学变量
    """
    # good_electron_mask = ak.zeros_like(events.event, dtype=bool)
    # good_muon_mask = ak.zeros_like(events.event, dtype=bool)
    
    # 安全处理电子
    if "Electron" in events.fields:
        try:
            electron_count = ak.num(events.Electron, axis=1)
            if ak.max(electron_count) > 0:
                e_pt = events.Electron.PT
                e_eta = events.Electron.Eta
                
                e_cuts = (e_pt > pt_cut) & (np.abs(e_eta) < eta_cut)
                good_electron_mask = e_cuts
                print(f"[GOODLEP] Processed {ak.sum(electron_count)} electrons successfully")
        except Exception as e:
            print(f"Skipping electrons due to error: {e}")
    
    # 安全处理缪子
    if "Muon" in events.fields:
        try:
            muon_count = ak.num(events.Muon, axis=1)
            if ak.max(muon_count) > 0:
                mu_pt = events.Muon.PT
                mu_eta = events.Muon.Eta
                
                mu_cuts = (mu_pt > pt_cut) & (np.abs(mu_eta) < eta_cut)
                good_muon_mask = mu_cuts
                print(f"[GOODLEP] Processed {ak.sum(muon_count)} muons successfully")
        except Exception as e:
            print(f"Skipping muons due to error: {e}")
    
    return good_electron_mask, good_muon_mask

def goodPhoton(events, pt_cut=10.0, eta_cut=2.5):
    """ select Photon with |eta| < etacut(2.5) and PT > ptcut(10.0) """
    Photon = ak.copy(events.Photon)
    exist_mask = ak.num(Photon, axis=1) > 0
    ptcut = Photon.PT > pt_cut           # PT > 0 GeV
    etacut = np.abs(Photon.Eta) < eta_cut  # |eta| < 2.5
    good_photon_cut = ptcut & etacut
    has_good_photon = ak.any(good_photon_cut, axis=1)
    mask = exist_mask & has_good_photon
    return mask
def missptOverCut(events, cut):
    # Check if MissingET.MET is greater than threshold
    mask = events["MissingET"].MET > cut
    return mask

def containBjet(events, num):
    #Check if events contain exactly the specified number of b-tagged jets
    Btag = events["Jet"].BTag
    btag_count = ak.sum(Btag == 1, axis=1)
    mask = btag_count == num
    return mask

def containOverBjet(events, num):
    #Check if events contain exactly the specified number of b-tagged jets
    Btag = events["Jet"].BTag
    btag_count = ak.sum(Btag == 1, axis=1)
    mask = btag_count >= num
    return mask

def countElectron(events):
    #Count the number of electrons in each event (both positive and negative charges)
    # Use ak.num to directly count electrons in each event
    electron_count = ak.num(events["Electron"], axis=1)
    return electron_count

def countMuon(events):
    #Count the number of muons in each event (both positive and negative charges)
    # Use ak.num to directly count muons in each event
    muon_count = ak.num(events["Muon"], axis=1)
    return muon_count

def leptonTrigger(events):
    #Check if events have exactly 1 lepton (electron + muon total = 1)
    electron_count = ak.num(events["Electron"], axis=1)
    muon_count = ak.num(events["Muon"], axis=1)
    total_lepton_count = electron_count + muon_count
    mask = total_lepton_count == 1
    return mask

def singleLeptonTrigger(events):
    # Check if events have exactly 1 good lepton (electron or muon)
    good_electron_mask, good_muon_mask = goodLepton(events, pt_cut=20.0)
    good_electron_count = ak.sum(good_electron_mask, axis=1)
    good_muon_count = ak.sum(good_muon_mask, axis=1)
    total_good_lepton_count = good_electron_count + good_muon_count
    mask = total_good_lepton_count == 1
    return mask

def tightElectron(events, cut):
    #Check if events have exactly 1 electron with PT > cut
    electron_pt = events["Electron"].PT
    # Count electrons with PT > cut in each event
    tight_electron_count = ak.sum(electron_pt > cut, axis=1)
    mask = tight_electron_count == 1
    return mask

def tightMuon(events, cut):
    # Check if events have exactly 1 muon with PT > cut
    muon_pt = events["Muon"].PT
    # Count muons with PT > cut in each event
    tight_muon_count = ak.sum(muon_pt > cut, axis=1)
    mask = tight_muon_count == 1
    return mask

print("[INIT] Functions loaded successfully")

def getWMass(events):
    missMET = events["MissingET"].MET
    missPhi = events["MissingET"].Phi
    ###
    # because we has limit lepton == 1, we just sum lepton PT and phi
    ###
    leptonPT = ak.sum(events["Electron"].PT, axis=1) + ak.sum(events["Muon"].PT, axis=1)
    leptonPhi = ak.sum(events["Electron"].Phi, axis=1) + ak.sum(events["Muon"].Phi, axis=1)
    ###
    # Reconstruct the transverse mass of W boson
    ###
    W_mass = np.sqrt(2 * leptonPT * missMET * (1 - np.cos(leptonPhi - missPhi)))
    return W_mass

def getBBmass(events):
    bjets = events["Jet"][events["Jet"].BTag == 1]
    bjets = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    leading2 = bjets[:, :2]
    px = leading2.PT * np.cos(leading2.Phi)
    py = leading2.PT * np.sin(leading2.Phi)
    pz = leading2.PT * np.sinh(leading2.Eta)
    E  = np.sqrt(leading2.Mass**2 + px**2 + py**2 + pz**2)
    px_sum = ak.sum(px, axis=1)
    py_sum = ak.sum(py, axis=1)
    pz_sum = ak.sum(pz, axis=1)
    E_sum  = ak.sum(E,  axis=1)
    mass2 = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    bbmass = np.sqrt(np.maximum(mass2, 0))  # 防止负数开根号
    return bbmass

def getBBmassValid(events):
    # select b-jets: BTag == 1, PT > 25, |eta| < 2.5
    bjets = events["Jet"][
        (events["Jet"].BTag == 1) & 
        (events["Jet"].PT > 25) & 
        (np.abs(events["Jet"].Eta) < 2.5)
    ]
    # 按 PT 降序排列（确保取到最高PT的b喷注）
    bjets = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    leading2 = bjets[:, :2]

    px = leading2.PT * np.cos(leading2.Phi)
    py = leading2.PT * np.sin(leading2.Phi)
    pz = leading2.PT * np.sinh(leading2.Eta)
    E = np.sqrt(leading2.Mass**2 + px**2 + py**2 + pz**2)

    px_sum = ak.sum(px, axis=1)
    py_sum = ak.sum(py, axis=1)
    pz_sum = ak.sum(pz, axis=1)
    E_sum = ak.sum(E, axis=1)

    mass2 = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    bb_mass = np.sqrt(np.maximum(mass2, 0))
    
    return bb_mass    

def getBBAmass(events):
    bjets = events["Jet"][events["Jet"].BTag == 1]
    bjets = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    leading2 = bjets[:, :2]
    photon = events["Photon"]
    px_a = photon.PT * np.cos(photon.Phi)
    py_a = photon.PT * np.sin(photon.Phi)
    pz_a = photon.PT * np.sinh(photon.Eta)
    E_a  = np.sqrt(px_a**2 + py_a**2 + pz_a**2)  
    px = leading2.PT * np.cos(leading2.Phi)
    py = leading2.PT * np.sin(leading2.Phi)
    pz = leading2.PT * np.sinh(leading2.Eta)
    E  = np.sqrt(leading2.Mass**2 + px**2 + py**2 + pz**2)
    px_sum = ak.sum(px, axis=1)
    py_sum = ak.sum(py, axis=1)
    pz_sum = ak.sum(pz, axis=1)
    E_sum  = ak.sum(E,  axis=1)
    # add photon four-vector components
    px_sum = px_sum + ak.sum(px_a, axis=1)  
    py_sum = py_sum + ak.sum(py_a, axis=1) 
    pz_sum = pz_sum + ak.sum(pz_a, axis=1)  
    E_sum  = E_sum + ak.sum(E_a, axis=1)    
    mass2 = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    bba_mass = np.sqrt(np.maximum(mass2, 0))  # 防止负数开根号
    return bba_mass

def getNjet(events):
    # get the number of jets in each event
    njet = ak.num(events["Jet"].BTag, axis=1)
    return njet
    
def fab(phi):
    # ensure phi is in the range [-π, π]
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
    return np.abs(phi)

def getDBjetDeltaSet(events):
    ## get LO two b jet: Delta R, Delta Eta, and Delta Phi
    bjets = events["Jet"][events["Jet"].BTag == 1]
    bjets = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    has_2bjets = ak.num(bjets, axis=1) >= 2
    bjets_padded = ak.pad_none(bjets, 2, axis=1)
    bjet1 = bjets_padded[:, 0]
    bjet2 = bjets_padded[:, 1]
    # Delta R
    delta_r = ak.where(
        has_2bjets,
        bjet1.delta_r(bjet2),
        np.nan
    )
    # Delta Eta
    delta_eta = ak.where(
        has_2bjets,
        np.abs(bjet1.Eta - bjet2.Eta),
        np.nan
    )
    # Delta Phi 
    delta_phi = ak.where(
        has_2bjets,
        fab(bjet1.Phi - bjet2.Phi),
        np.nan
    )
    return delta_r, delta_eta, delta_phi

def getPhotonEnergy(events, return_type='leading'):
    ## get Photon energy based on the specified return type
    photons = events["Photon"]
    
    if return_type == 'all':
        return photons.E
    elif return_type == 'leading':
        photons_sorted = photons[ak.argsort(photons.PT, axis=1, ascending=False)]
        has_photons = ak.num(photons_sorted, axis=1) > 0
        photons_padded = ak.pad_none(photons_sorted, 1, axis=1)
        return ak.where(
            has_photons,
            photons_padded[:, 0].E,
            np.nan
        )
    else:
        total_energy = ak.sum(photons.E, axis=1)
        return ak.fill_none(total_energy, 0.0)
    
def getPhotonPT(events, return_type='leading'):
    ## get Photon PT based on the specified return type
    photons = events["Photon"]
    
    if return_type == 'all':
        return photons.PT
    elif return_type == 'leading':
        photons_sorted = photons[ak.argsort(photons.PT, axis=1, ascending=False)]
        has_photons = ak.num(photons_sorted, axis=1) > 0
        photons_padded = ak.pad_none(photons_sorted, 1, axis=1)
        return ak.where(
            has_photons,
            photons_padded[:, 0].PT,
            np.nan
        )
    else:
        total_pt = ak.sum(photons.PT, axis=1)
        return ak.fill_none(total_pt, 0.0)

def getHiggsFromDBjet(events):
    ## reconstruct Higgs boson from two leading b-jets
    bjets = events["Jet"][events["Jet"].BTag == 1]
    bjets = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    has_2bjets = ak.num(bjets, axis=1) >= 2
    leading2 = ak.pad_none(bjets, 2, axis=1)[:, :2]
    
    px = leading2.PT * np.cos(leading2.Phi)
    py = leading2.PT * np.sin(leading2.Phi)
    pz = leading2.PT * np.sinh(leading2.Eta)
    E  = np.sqrt(leading2.Mass**2 + px**2 + py**2 + pz**2)
    
    px_sum = ak.sum(px, axis=1)
    py_sum = ak.sum(py, axis=1)
    pz_sum = ak.sum(pz, axis=1)
    E_sum  = ak.sum(E,  axis=1)

    mass2 = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    higgs_mass = np.sqrt(np.maximum(mass2, 0))
    higgs_pt = np.sqrt(px_sum**2 + py_sum**2)
    higgs_eta = np.arcsinh(pz_sum / higgs_pt)
    higgs_phi = np.arctan2(py_sum, px_sum)

    higgs = ak.zip({
        "PT": ak.where(has_2bjets, higgs_pt, np.nan),
        "Eta": ak.where(has_2bjets, higgs_eta, np.nan),
        "Phi": ak.where(has_2bjets, higgs_phi, np.nan),
        "Mass": ak.where(has_2bjets, higgs_mass, np.nan),
        "E": ak.where(has_2bjets, E_sum, np.nan),
        "PX": ak.where(has_2bjets, px_sum, np.nan),
        "PY": ak.where(has_2bjets, py_sum, np.nan),
        "PZ": ak.where(has_2bjets, pz_sum, np.nan)
    }, with_name="Momentum4D")
    
    return higgs
def deltaR(obj1_eta, obj1_phi, obj2_eta, obj2_phi):
    """Calculate delta R between two objects"""
    delta_eta = obj1_eta - obj2_eta
    delta_phi = fab(obj1_phi - obj2_phi)
    
    return np.sqrt(delta_eta**2 + delta_phi**2)
def getHiggsPhotonDeltaR(events):
    ## calculate the delta R between Higgs and leading photon
    higgs = getHiggsFromDBjet(events)
    photons = events["Photon"]

    has_photons = ak.num(photons, axis=1) > 0
    has_higgs = ~ak.is_none(higgs.PT) & ~np.isnan(higgs.PT)

    photons_padded = ak.pad_none(photons, 1, axis=1)
    photon1 = photons_padded[:, 0]
    delta_r = ak.where(
        has_photons & has_higgs,
        deltaR(higgs.Eta, higgs.Phi, photon1.Eta, photon1.Phi),
        np.nan
    )
    return delta_r

def getLeptonPhotonDeltaR(events):
    """
    Calculate delta R between leading lepton and leading photon
    """
    # Get leading lepton (electron or muon)
    electrons = events["Electron"]
    muons = events["Muon"]
    
    has_electrons = ak.num(electrons, axis=1) > 0
    has_muons = ak.num(muons, axis=1) > 0
    
    leading_electron = ak.pad_none(electrons, 1, axis=1)[:, 0]
    leading_muon = ak.pad_none(muons, 1, axis=1)[:, 0]
    
    # Get leading photon
    photons = events["Photon"]
    has_photons = ak.num(photons, axis=1) > 0
    leading_photon = ak.pad_none(photons, 1, axis=1)[:, 0]
    
    # Calculate delta R for electron and photon
    delta_r_electron = ak.where(
        has_electrons & has_photons,
        deltaR(leading_electron.Eta, leading_electron.Phi, leading_photon.Eta, leading_photon.Phi),
        np.nan
    )
    
    # Calculate delta R for muon and photon
    delta_r_muon = ak.where(
        has_muons & has_photons,
        deltaR(leading_muon.Eta, leading_muon.Phi, leading_photon.Eta, leading_photon.Phi),
        np.nan
    )
    
    # Combine: use electron if available, otherwise use muon
    delta_r_lepton = ak.where(
        has_electrons,
        delta_r_electron,
        delta_r_muon
    )
    
    return delta_r_lepton


def getPhotonBjetDeltaR(events, mode='leading'):
    # Filter b-tagged jets
    bjets = events["Jet"][events["Jet"].BTag == 1]
    photons = events["Photon"]
    # Check if events have required objects
    has_bjets = ak.num(bjets, axis=1) > 0
    has_photons = ak.num(photons, axis=1) > 0
    valid_events = has_bjets & has_photons
    if mode == 'leading':
        # Sort by PT and take leading objects
        bjets_sorted = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
        photons_sorted = photons[ak.argsort(photons.PT, axis=1, ascending=False)]
        
        leading_bjet = ak.pad_none(bjets_sorted, 1, axis=1)[:, 0]
        leading_photon = ak.pad_none(photons_sorted, 1, axis=1)[:, 0]
        
        delta_r = ak.where(
            valid_events,
            deltaR(leading_bjet.Eta, leading_bjet.Phi, 
                   leading_photon.Eta, leading_photon.Phi),
            np.nan
        )
    elif mode == 'closest':
        # Find closest b-jet to leading photon
        photons_sorted = photons[ak.argsort(photons.PT, axis=1, ascending=False)]
        leading_photon = ak.pad_none(photons_sorted, 1, axis=1)[:, 0]
        
        # Calculate delta R to all b-jets
        photon_eta = ak.broadcast_arrays(leading_photon.Eta, bjets.Eta)[0]
        photon_phi = ak.broadcast_arrays(leading_photon.Phi, bjets.Phi)[0]
        
        all_dr = deltaR(bjets.Eta, bjets.Phi, photon_eta, photon_phi)
        min_dr = ak.min(all_dr, axis=1)
        
        delta_r = ak.where(valid_events, min_dr, np.nan)
    elif mode == 'min':
        # Minimum delta R among all photon-bjet pairs
        photon_eta, bjet_eta = ak.unzip(ak.cartesian([photons.Eta, bjets.Eta], nested=True))
        photon_phi, bjet_phi = ak.unzip(ak.cartesian([photons.Phi, bjets.Phi], nested=True))
        
        all_dr = deltaR(bjet_eta, bjet_phi, photon_eta, photon_phi)
        min_dr = ak.min(ak.min(all_dr, axis=2), axis=1)
        
        delta_r = ak.where(valid_events, min_dr, np.nan)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'leading', 'closest', or 'min'")
    
    return delta_r

def getBjetWbosonFakeMass(events):
    """
    Calculate the invariant mass of b-jet + lepton + MET (top quark reconstruction)
    
    This function reconstructs the top quark mass by combining:
    - Leading b-tagged jet
    - Leading lepton (electron or muon)
    - Missing transverse energy (representing neutrino)
    """
    import numpy as np
    import awkward as ak
    
    # Get leading b-jet
    bjets = events["Jet"][events["Jet"].BTag == 1]
    has_bjets = ak.num(bjets, axis=1) > 0
    bjets_sorted = bjets[ak.argsort(bjets.PT, axis=1, ascending=False)]
    leading_bjet = ak.pad_none(bjets_sorted, 1, axis=1)[:, 0]
    
    # Get leading lepton
    electrons = ak.pad_none(events["Electron"], 1, axis=1)[:, 0]
    muons = ak.pad_none(events["Muon"], 1, axis=1)[:, 0]
    
    electron_pt = ak.fill_none(electrons.PT, 0)
    muon_pt = ak.fill_none(muons.PT, 0)
    
    is_electron = electron_pt > muon_pt
    leading_lepton = ak.where(is_electron, electrons, muons)
    
    # Lepton mass (in GeV)
    lepton_mass = ak.where(is_electron, 0.000511, 0.105658)
    
    # Get MissingET
    missing_et = events["MissingET"]
    
    # Calculate four-vectors
    # B-jet
    bjet_px = leading_bjet.PT * np.cos(leading_bjet.Phi)
    bjet_py = leading_bjet.PT * np.sin(leading_bjet.Phi)
    bjet_pz = leading_bjet.PT * np.sinh(leading_bjet.Eta)
    bjet_E = np.sqrt(leading_bjet.Mass**2 + bjet_px**2 + bjet_py**2 + bjet_pz**2)
    
    # Lepton
    lepton_px = leading_lepton.PT * np.cos(leading_lepton.Phi)
    lepton_py = leading_lepton.PT * np.sin(leading_lepton.Phi)
    lepton_pz = leading_lepton.PT * np.sinh(leading_lepton.Eta)
    lepton_E = np.sqrt(lepton_mass**2 + lepton_px**2 + lepton_py**2 + lepton_pz**2)
    
    # MET (assuming massless neutrino)
    met_px = missing_et.MET * np.cos(missing_et.Phi)
    met_py = missing_et.MET * np.sin(missing_et.Phi)
    met_pz = 0.0  # MET is in transverse plane
    met_E = missing_et.MET  # For massless particle, E = |p|
    
    # Calculate invariant mass
    total_E = bjet_E + lepton_E + met_E
    total_px = bjet_px + lepton_px + met_px
    total_py = bjet_py + lepton_py + met_py
    total_pz = bjet_pz + lepton_pz + met_pz
    
    invariant_mass_squared = total_E**2 - (total_px**2 + total_py**2 + total_pz**2)
    invariant_mass = np.sqrt(np.maximum(invariant_mass_squared, 0))
    
    # Only return valid masses (events with b-jets and leptons)
    has_lepton = (electron_pt > 0) | (muon_pt > 0)
    valid_events = has_bjets & has_lepton
    
    return ak.where(valid_events, invariant_mass, np.nan)

print("[INIT] Utility functions loaded successfully!") 


def eventCut(events, name="default"):
    maskPhoton = goodPhoton(events, pt_cut=20)
    maskmisspt = missptOverCut(events, 30)
    maskBjets = containOverBjet(events, 2)
    maskLepTrig = leptonTrigger(events)
    maskSingleLepTrig = singleLeptonTrigger(events)
    
    maskElec = tightElectron(events, 30)
    maskMuon = tightMuon(events, 26)
    maskLepPT = mergeOrMasks(maskElec, maskMuon)
    mask_list = (
        maskPhoton, 
        maskmisspt, 
        maskBjets, 
        maskSingleLepTrig,
        # maskLepTrig, 
        # maskLepPT
    )
    merged_mask = mergeAndMasks(mask_list)

    for mask in mask_list:
        ## return efficiency of each mask
        efficiency = getEfficiency(mask, total_events=len(events))
        print(f"[CUTTER] {name} Mask efficiency: {efficiency:.2%}")

    # cutnum = len(events[merged_mask])
    text= f"[CUTTER] {name} Before cut: {len(events)} events, After cut: {len(events[merged_mask])} events"
    print(text)
    return merged_mask

def analyzeEvents(events, mask, cut_name="default", path=""):
    ## Analyze events and return results
    print(f"[ANALYZE] Analyzing events with cut: {cut_name}")
    
    # 应用掩码
    events_cut = events[mask]
    n_events = len(events_cut)
    
    if n_events == 0:
        print(f"[WARNING] No events pass the cut: {cut_name}")
        return None
    
    # 计算各种物理量
    w_mass = getWMass(events_cut)
    bbMass = getBBmass(events_cut)
    bbaMass = getBBAmass(events_cut)
    njet = getNjet(events_cut)
    dB_delta_r, dB_delta_eta, dB_delta_phi = getDBjetDeltaSet(events_cut)
    bP_delta_r = getHiggsPhotonDeltaR(events_cut)
    photon_energy = getPhotonEnergy(events_cut, return_type='leading')
    
    photon_pt = getPhotonPT(events_cut, return_type='leading')
    aB_delta_r = getPhotonBjetDeltaR(events_cut, mode='leading')
    aL_delta_r_lepton = getLeptonPhotonDeltaR(events_cut)
    fake_top_mass = getBjetWbosonFakeMass(events_cut)

    print(f"[ANALYZE] Analyzed {n_events} events with cut: {cut_name} Done")
    
    result = {
        "W_mass": w_mass,
        "BB_mass": bbMass,
        "BBA_mass": bbaMass,
        "Njet": njet,
        "dB_delta_r": dB_delta_r,
        "dB_delta_eta": dB_delta_eta,
        "dB_delta_phi": dB_delta_phi,
        "bP_delta_r": bP_delta_r,
        "photon_energy": photon_energy,
        "photon_pt": photon_pt,
        "aB_delta_r": aB_delta_r,
        "aL_delta_r_lepton": aL_delta_r_lepton,
        "fake_top_mass": fake_top_mass
    }
    
    # DataFrame
    # df = saveResultsEBE(result, cut_name=cut_name, path=path)
    return result

def dBjetDeltaCut(events, cut, name="default"):
    """
    Apply delta R cut on b-jets and return the mask.
    cut = [deltaR cut, deltaEta cut, deltaPhi cut]
    """
    # 检查每个 event 是否至少有 2 个 b-jet
    Btag = events["Jet"].BTag
    n_bjets = ak.sum(Btag == 1, axis=1)
    has_two_bjets = n_bjets >= 2
    
    # 初始化所有 event 的 mask 为 False
    mask = ak.zeros_like(has_two_bjets, dtype=bool)
    
    # 只对包含至少 2 个 b-jet 的 event 进行计算
    if ak.sum(has_two_bjets) > 0:  # 如果存在有 2 个以上 b-jet 的 event
        events_with_bjets = events[has_two_bjets]
        
        # 计算这些 event 的 delta 值
        dB_delta_r, dB_delta_eta, dB_delta_phi = getDBjetDeltaSet(events_with_bjets)
        
        # 应用 cut
        maskdeltar = dB_delta_r < cut[0]
        maskdeltaEta = dB_delta_eta < cut[1]
        maskdeltaPhi = dB_delta_phi < cut[2]
        
        # 合并 mask
        cut_result = mergeAndMasks(maskdeltar, maskdeltaEta, maskdeltaPhi)
        
        # 将结果填回到原始长度的 mask 中
        # 转换为numpy数组进行索引操作
        mask_np = ak.to_numpy(mask)
        has_two_bjets_np = ak.to_numpy(has_two_bjets)
        cut_result_np = ak.to_numpy(cut_result)
        
        mask_np[has_two_bjets_np] = cut_result_np
        mask = ak.Array(mask_np)
    
    return mask




def eventCutL2(events, mask, name="default"):
    """
    Apply cut and return the mask
    """
    print(f"[CUTTER-L2] Applying cut: {name}")
    dbjetDeltaCut_mask = dBjetDeltaCut(events, cut=[4, 3, 2], name=name)
    # 将新的 cut mask 与输入的 mask 合并
    combined_mask = mask & dbjetDeltaCut_mask
    print(f"[CUTTER-L2] {name} Before L2 cut: {ak.sum(mask==True)} events, After L2 cut: {ak.sum(combined_mask==True)} events")
    
    return combined_mask


def saveResultsEBE(result, cut_name="default", path=""):
    """将结果保存为 Pandas DataFrame"""
    print(f"[SAVE] Converting results to DataFrame...")
    # if no result, return empty DataFrame
    if result is None or len(result) == 0:
        print(f"[SAVE] No results to save for cut: {cut_name}")
        return pd.DataFrame()
    df_data = {}
    for key, value in result.items():
        try:
            if hasattr(value, 'to_numpy'):
                np_array = ak.to_numpy(value, allow_missing=True)
            else:
                np_array = np.asarray(value)
            if np_array.ndim > 1:
                np_array = np_array.flatten()
            df_data[key] = np_array
            
        except Exception as e:
            print(f"[WARNING] Could not convert {key}: {e}")
            continue
    
    # DataFrame
    df = pd.DataFrame(df_data)
    if path:
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"analysis_{cut_name}.parquet")
    else:
        filename = f"analysis_{cut_name}.parquet"
    
    df.to_parquet(filename)
    print(f"[SAVE] DataFrame saved to: {filename}")
    print(f"[SAVE] Shape: {df.shape}, Columns: {list(df.columns)}")
    
    return df

print("[INIT] Analysis functions loaded successfully!")

print("[SYSTEM] Starting background analysis...")
signal_events = readDelphes("samples/whg_lo_pp_vlh_decay_dd.root")
signal2_events = readDelphes("samples/non_higgs_signal2.root")
bkg1_events = readDelphes("samples/whgbkg1_p.root")
bkg2_events = readDelphes("samples/whgbkg2_p.root")
bkg3_events = readDelphes("samples/whgbkg3_p.root")
bkg4_events = readDelphes("samples/whgbkg4_p.root")

print("[SYSTEM] Background events loaded successfully!")
print("[SYSTEM] Starting event selection...")
signal_mask = eventCut(signal_events, name="Signal")
signal2_mask = eventCut(signal2_events, name="Signal2")
bkg1_mask = eventCut(bkg1_events, name="Bkg1")
bkg2_mask = eventCut(bkg2_events, name="Bkg2")
bkg3_mask = eventCut(bkg3_events, name="Bkg3")
bkg4_mask = eventCut(bkg4_events, name="Bkg4")
print("[SYSTEM] Event selection completed!")
print("[SYSTEM] Starting event analysis...")
signal_result = analyzeEvents(signal_events, signal_mask, cut_name="Signal", path="data")
signal2_result = analyzeEvents(signal2_events, signal2_mask, cut_name="Signal2", path="data")
bkg1_result = analyzeEvents(bkg1_events, bkg1_mask, cut_name="Bkg1", path="data")
bkg2_result = analyzeEvents(bkg2_events, bkg2_mask, cut_name="Bkg2", path="data")
bkg3_result = analyzeEvents(bkg3_events, bkg3_mask, cut_name="Bkg3", path="data")
bkg4_result = analyzeEvents(bkg4_events, bkg4_mask, cut_name="Bkg4", path="data")
print("[SYSTEM] Event analysis completed!")
print("[SYSTEM] Saving results...")
signal_L2_mask = eventCutL2(signal_events, signal_mask,name="Signal")
signal2_L2_mask = eventCutL2(signal2_events, signal2_mask,name="Signal2")
bkg1_L2_mask = eventCutL2(bkg1_events, bkg1_mask,name="Bkg1")
bkg2_L2_mask = eventCutL2(bkg2_events, bkg2_mask,name="Bkg2")
bkg3_L2_mask = eventCutL2(bkg3_events, bkg3_mask,name="Bkg3")
bkg4_L2_mask = eventCutL2(bkg4_events, bkg4_mask,name="Bkg4")
print("[SYSTEM] L2 event selection completed!")
print("[SYSTEM] Analyzing L2 events...")
signal_L2_result = analyzeEvents(signal_events, signal_L2_mask, cut_name="Signal_L2", path="data")
signal2_L2_result = analyzeEvents(signal2_events, signal2_L2_mask, cut_name="Signal2_L2", path="data")
bkg1_L2_result = analyzeEvents(bkg1_events, bkg1_L2_mask, cut_name="Bkg1_L2", path="data")
bkg2_L2_result = analyzeEvents(bkg2_events, bkg2_L2_mask, cut_name="Bkg2_L2", path="data")
bkg3_L2_result = analyzeEvents(bkg3_events, bkg3_L2_mask, cut_name="Bkg3_L2", path="data")
bkg4_L2_result = analyzeEvents(bkg4_events, bkg4_L2_mask, cut_name="Bkg4_L2", path="data")
print("[SYSTEM] L2 event analysis completed!")
print("[SYSTEM] Saving results...")
saveResultsEBE(signal_L2_result, cut_name="Signal_L2", path="data")
saveResultsEBE(signal2_L2_result, cut_name="Signal2_L2", path="data")
saveResultsEBE(bkg1_L2_result, cut_name="Bkg1_L2", path="data")
saveResultsEBE(bkg2_L2_result, cut_name="Bkg2_L2", path="data")
saveResultsEBE(bkg3_L2_result, cut_name="Bkg3_L2", path="data")
saveResultsEBE(bkg4_L2_result, cut_name="Bkg4_L2", path="data")
print("[SYSTEM] L2 results saved successfully!")
print("[SYSTEM] Saving initial results...")
saveResultsEBE(signal_result, cut_name="Signal", path="data")
saveResultsEBE(signal2_result, cut_name="Signal2", path="data")
saveResultsEBE(bkg1_result, cut_name="Bkg1", path="data")
saveResultsEBE(bkg2_result, cut_name="Bkg2", path="data")
saveResultsEBE(bkg3_result, cut_name="Bkg3", path="data")
saveResultsEBE(bkg4_result, cut_name="Bkg4", path="data")
print("[SYSTEM] Results saved successfully!")
print("[SYSTEM] Exiting background analysis script...")