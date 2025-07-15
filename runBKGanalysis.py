#!/anaconda3/envs/oldcoffeaenv/bin/python
import uproot
import numpy as np
import matplotlib.pyplot as plt
# import coffea

import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, DelphesSchema


def readDelphes(name):
    events = NanoEventsFactory.from_root(
        name,
        schemaclass=DelphesSchema,
        treepath="Delphes",
    ).events()
    return events

def countItem(arr, item):
    count = 0
    for i in arr:
        if i == item:
            count += 1
    return count
def mergeAddMasks(*masks):
    merged_mask = masks[0]
    for mask in masks[1:]:
        merged_mask = merged_mask & mask
    return merged_mask
def mergeOrMasks(*masks):
    merged_mask = masks[0]
    for mask in masks[1:]:
        merged_mask = merged_mask or mask
    return merged_mask
def mergeXorMasks(mask1, mask2):
    merged_mask = mask1 ^ mask2
    return merged_mask

def missptOverCut(events, cut):
    mask = events["MissingET"].MET > cut
    return mask
def containBjet(events,num):
    ### loose = bool (a & 1 )  medium = bool (a & 2) tight = bool(a & 4) 
    Btag = events["Jet"].BTag
    mask = []
    for i in Btag:
        if countItem(i, 1) >= num:
            mask.append(True)
        else:
            mask.append(False)
    return mask
def countElectron(events):
    Electron = events["Electron"].Charge
    electron_count = []
    for i in Electron:
        electron_num = countItem(i, 1)
        negative_num = countItem(i, -1)
        num = electron_num + negative_num
        electron_count.append(num)
    return electron_count
def countMuon(events):
    Muon = events["Muon"].Charge
    muon_count = []
    for i in Muon:
        muon_num = countItem(i, 1)
        negative_num = countItem(i, -1)
        num = muon_num + negative_num
        muon_count.append(num)
    return muon_count
def leptonTrigger(events):
    electron_count = countElectron(events)
    muon_count = countMuon(events)
    # lepton = electron_count + muon_count
    lepton = []
    for i in range(len(events)):
        leptons = electron_count[i] + muon_count[i]
        lepton.append(leptons)
    mask = []
    for i in lepton:
        if i == 1:
            mask.append(True)
        else:
            mask.append(False)
    return mask
def tightElectron(events, cut):
    Electron = events["Electron"].PT
    electron_count = []
    for i in range(len(Electron)):
        count = 0
        for j in Electron[i]:
            if j > cut:
                count += 1
        electron_count.append(count)
    mask = []
    for i in electron_count:
        if i == 1:
            mask.append(True)
        else:
            mask.append(False)
    return mask
def tightMuon(events, cut):
    Muon = events["Muon"].PT
    muon_count = []
    for i in range(len(Muon)):
        count = 0
        for j in Muon[i]:
            if j > cut:
                count += 1
        muon_count.append(count)
    mask = []
    for i in muon_count:
        if i == 1:
            mask.append(True)
        else:
            mask.append(False)
    return mask

# good lepton function
def goodLepton(events):
    # select Electron and Muon
    # with |eta| < 2.5
    mask=[]
    return mask
def goodPhoton(events):

print("Functions loaded successfully!")


def eventCut(events):
    maskmisspt = missptOverCut(events, 30)  # 72%, 64%, 62%, 29%, 48%
    maskBjets = containBjet(events, 2)  # 0.0158%, 33.8%, 17%, 1.3%, 1.4%
    maskLepTrig = leptonTrigger(events) # 82%, 22%, 25%, 11% 21%
    
    maskElec = tightElectron(events, 30)    #10
    maskMuon = tightMuon(events, 26)        #10
    maskLepPT = mergeOrMasks(maskElec, maskMuon)
    maskchoose = (
        maskmisspt,
        maskBjets,
        maskLepTrig,
        # maskLepPT
    )

    merged_mask = mergeAddMasks(*maskchoose)
    cutnum = countItem(merged_mask, True)
    text= f"Before cut: {len(events)} events, After cut: {len(events[merged_mask])} events"
    print(text)
    return merged_mask

# events = events[merged_mask]
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
    
    ### because we has limit bjet == 2,
    ### loose = bool (a & 1 )  medium = bool (a & 2) tight = bool(a & 4) 
    # bjetmask = events["Jet"].BTag == 1 
    # bjet_1_pt = events["Jet"].PT[bjetmask][:, 0]
    # bjet_2_pt = events["Jet"].PT[bjetmask][:, 1]
    # bjet_1_eta = events["Jet"].Eta[bjetmask][:, 0]
    # bjet_2_eta = events["Jet"].Eta[bjetmask][:, 1]
    # bjet_1_phi = events["Jet"].Phi[bjetmask][:, 0]
    # bjet_2_phi = events["Jet"].Phi[bjetmask][:, 1]
    # bjet_1_mass = events["Jet"].Mass[bjetmask][:, 0]
    # bjet_2_mass = events["Jet"].Mass[bjetmask][:, 1]    
    bjet_1_pt = events["Jet"].PT[:, 0]
    bjet_2_pt = events["Jet"].PT[:, 1]
    bjet_1_eta = events["Jet"].Eta[:, 0]
    bjet_2_eta = events["Jet"].Eta[:, 1]
    bjet_1_phi = events["Jet"].Phi[:, 0]
    bjet_2_phi = events["Jet"].Phi[:, 1]
    bjet_1_mass = events["Jet"].Mass[:, 0]
    bjet_2_mass = events["Jet"].Mass[:, 1]
    ### calculate the mass
    bjet_mass = bjet_1_mass**2 + bjet_2_mass**2 +np.sqrt(
        2 * bjet_1_pt * bjet_2_pt * 
        (np.cosh(bjet_1_eta - bjet_2_eta) - np.cos(bjet_1_phi - bjet_2_phi))
    )
    bjet_mass = np.sqrt(bjet_mass)


    return bjet_mass
def saveAsNPY(events, filename):
    """
    Save the events as a numpy file.
    """
    np.save(filename, ak.to_numpy(events))
    print(f"Events saved to {filename}")

print("CUT ready!")

print("Reading Delphes file...")
signal_events = readDelphes("bkgdelphes/whg_lo_pp_vlh_decay_dd.root")
bkg1_events = readDelphes("bkgdelphes/whgbkg1_p.root")
bkg2_events = readDelphes("bkgdelphes/whgbkg2_p.root")
bkg3_events = readDelphes("bkgdelphes/whgbkg3_p.root")
bkg4_events = readDelphes("bkgdelphes/whgbkg4_p.root")

print("cutting events...")
signal_mask = eventCut(signal_events)
signal_cut = signal_events[signal_mask]

bkg1_mask = eventCut(bkg1_events)
bkg1_cut = bkg1_events[bkg1_mask]
bkg2_mask = eventCut(bkg2_events)
bkg2_cut = bkg2_events[bkg2_mask]
bkg3_mask = eventCut(bkg3_events)
bkg3_cut = bkg3_events[bkg3_mask]
bkg4_mask = eventCut(bkg4_events)
bkg4_cut = bkg4_events[bkg4_mask]


print("starting Analysis...")
signal_mass = getWMass(signal_cut)
bkg1_mass = getWMass(bkg1_cut)
bkg2_mass = getWMass(bkg2_cut)
bkg3_mass = getWMass(bkg3_cut)
bkg4_mass = getWMass(bkg4_cut) 

signal_bb_mass = getBBmass(signal_cut)
bkg1_bb_mass = getBBmass(bkg1_cut)
bkg2_bb_mass = getBBmass(bkg2_cut)
bkg3_bb_mass = getBBmass(bkg3_cut)
bkg4_bb_mass = getBBmass(bkg4_cut)
print("Analysis complete!")
print("saving results...")
saveAsNPY(signal_mass, "bkgdelphes/npy/signal_mass.npy")
saveAsNPY(bkg1_mass, "bkgdelphes/npy/bkg1_mass.npy")
saveAsNPY(bkg2_mass, "bkgdelphes/npy/bkg2_mass.npy")
saveAsNPY(bkg3_mass, "bkgdelphes/npy/bkg3_mass.npy")
saveAsNPY(bkg4_mass, "bkgdelphes/npy/bkg4_mass.npy")
saveAsNPY(signal_bb_mass, "bkgdelphes/npy/signal_bb_mass.npy")
saveAsNPY(bkg1_bb_mass, "bkgdelphes/npy/bkg1_bb_mass.npy")
saveAsNPY(bkg2_bb_mass, "bkgdelphes/npy/bkg2_bb_mass.npy")
saveAsNPY(bkg3_bb_mass, "bkgdelphes/npy/bkg3_bb_mass.npy")
saveAsNPY(bkg4_bb_mass, "bkgdelphes/npy/bkg4_bb_mass.npy")
print("Results saved successfully!")
print("Exit")
