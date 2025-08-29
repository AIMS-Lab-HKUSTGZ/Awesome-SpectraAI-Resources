# Awesome-SpectraAI-Resources
<div align="center">
  <img width="690" height="83" alt="Awesome SpectraAI Resources" src="https://github.com/user-attachments/assets/5ecb4f23-cb12-46c8-bd8b-5f13481a5850" />
</div>

✨✨ A curated collection of resources on artificial intelligence for spectral data analysis, covering computational methods for mass spectrometry (MS), NMR, IR, and XRD data.

---

## 1. Mass Spectrometry (Small Molecules)

### 1.1 Forward Task (Molecule → Spectrum)  
*Computational approaches for predicting mass spectra from molecular structures*  
 

### 1.2 Inverse Task (Spectrum → Molecule)  
*AI methods for molecular identification and elucidation from mass spectra*  


---

## 2. Mass Spectrometry (Peptides)

### 2.1 Forward Task (Peptides → Spectrum)  
*Computational methods for predicting peptides mass spectra*  


### 2.2 Inverse Task (Spectrum → Peptides)  
*AI approaches for peptides identification and quantification*  


---

## 3. NMR Spectroscopy (Small Molecules)

### 3.1 Forward Task (Molecule → Spectrum)  
*Prediction of NMR spectra from molecular structures*  
#### 📊 Forward Task Method Table
| Paper Title & Link | Method Type | Data Source | Performance Metric | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [Prediction of chemical shift in NMR: A review](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/mrc.5234) | Empirical | - | Rule-based | Interpretable, less generalizable |
| [iShiftML: Highly Accurate Prediction of NMR Chemical Shifts](https://arxiv.org/abs/2306.08269) | Hybrid ML + QM | QM descriptors | < 0.2 ppm error | Fast inference, needs QM feature prep |
| [NMR shift prediction from small data quantities](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00785-x) | ML | NMRShiftDB2 | MAE (ppm) | Good scalability |
| [NMR-spectrum prediction for dynamic molecules](https://pubs.aip.org/aip/jcp/article/158/19/194108/2891394) | ML-Dynamics | Simulated ensembles | Time-avg ppm | Accounts for flexible molecules |
| [Machine learning in NMR spectroscopy](https://www.sciencedirect.com/science/article/pii/S0079656525000196) | DL | NMRShiftDB2 | TBD | Multitask joint learning |


---


### 3.2 Inverse Task: Spectrum → Molecule
#### 📊 Inverse Task Method Table
| Paper Title & Link | Method Type | Input Data | Accuracy / Metric | Notes |
|--------------------|-------------|------------|------------------|-------|
| [A Bayesian approach to structural elucidation using crystalline-state solid‑state NMR and probabilistic inference (2019)](https://arxiv.org/abs/1909.00870) | Bayesian | Solid‑state NMR | Top‑5 accuracy | Requires crystal information |
| [Accurate and efficient structure elucidation from routine one‑dimensional NMR spectra using multitask machine learning (2024)](https://arxiv.org/abs/2408.08284) | DL (CNN + Transformer) | 1D spectra | Top‑1 ~70% | No need for 2D spectra |
| [Deep reinforcement learning and graph convolutional networks for molecular inverse problem of NMR (2022)](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c00624) | RL (MCTS + GCN) | Shift table | Top‑3 ~80% | Effective for small molecules |
| [High‑resolution iterative Full Spin Analysis (HiFSA) for small molecules using PERCH (2015)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3812940/) | Spectral ID | Simulated spectra | — | Useful for detailed peak assignment |
| [Automated mixture component identification via wavelet packet transform and optimization (2023)](https://www.mdpi.com/1420-3049/28/2/792) | Mixture ID (WPT + Optimization) | Mixtures | Component-level accuracy | Robust for complex sample spectra | 

### 🧬 NMR Dataset Comparison Table


| Dataset Name & Link | Spectrum Count | Real / Simulated | Multi-modal Spectra | Labeled | Downloadable / Crawlable |
|---------------------|----------------|------------------|----------------------|---------|---------------------------|
| [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/) | ~50,000 | Real | ¹H, ¹³C | ✅ Yes | ✅ Yes (open source) |
| [BMRB](https://bmrb.io/) | >13,000 biomolecules | Real | ¹H, ¹³C, ¹⁵N, ²H, ³¹P | ✅ Yes | ✅ Yes (FTP/STAR) |
| [SDBS](https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_index.cgi) | ~14,000 | Real | ¹H, ¹³C, IR, MS, UV | ✅ Yes | ✅ Yes (Crawl Script Needed) |
| [QM9-NMR (Simulated)](https://doi.org/10.1021/acs.jcim.1c01160) | 130,000+ | Simulated (DFT) | ¹H, ¹³C | ✅ Yes | ✅ Yes (via DOI or GitHub) |
| [2DNMRGym (2024)](https://arxiv.org/abs/2405.18181) | 22,000 2D HSQC | Simulated | HSQC (2D) | ✅ Yes | ✅ Yes (HuggingFace) |
| [NMRMixDB](https://nmrmixdb.github.io/) | ~3,000 mixtures | Real | ¹H | ✅ Yes (with labels) | ✅ Yes |
| [NMRPredBench](https://github.com/ur-whitelab/NMRPredBench) | ~3,000 | Real + Simulated | ¹H, ¹³C | ✅ Yes | ✅ Yes (GitHub) |
[MolAid](https://mol.org/) | ~840K+ | Experimental | Multi-property | ✅ Yes |  ❌ No(API Chared) | Chinese chemical big data platform |
| [NIST WebBook](https://webbook.nist.gov/) | ~700K+ | Experimental | ¹H, ¹³C etc. | ✅ Yes | ✅ Yes (Need Search Key) | NIST-standardized spectral database |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | ~100M+ | Experimental + Predicted | Full compound attributes | ✅ Yes | ✅ Yes (API) | Largest open chemical database |
---


### 📚 Synthetic NMR Papers & Datasets


| Category                 | Name                                 | What it offers                                                       | Typical Use                                                  | Link                                                                                                     |
| ------------------------ | ------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Dataset & Pipeline       | **ARTINA – 100‑Protein NMR Dataset** | 1329 2D–4D spectra + chemical shift assignments + protein structures | Train/test protein NMR auto‑assignment & structure inference | [https://www.nature.com/articles/s41597-023-02879-5](https://www.nature.com/articles/s41597-023-02879-5) |
| Database (LLM‑extracted) | **NMRBank**                          | \~225k small‑molecule records (SMILES, ¹H/¹³C shifts)                | Build ML models for shift/spectrum prediction                | [https://pmc.ncbi.nlm.nih.gov/articles/PMC12118362/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12118362/) |
| Synthetic multimodal     | **IR‑NMR Dataset**                   | 177k IR spectra + 1.2k NMR shifts (DFT+ML)                           | Cross‑modal learning / pretraining                           | [https://www.nature.com/articles/s41597-025-05729-8](https://www.nature.com/articles/s41597-025-05729-8) |
| Carbohydrates            | **GlycoNMR**                         | 2,609 glycans with 211k NMR shifts                                   | Domain‑specific ML (carbohydrates)                           | [https://arxiv.org/abs/2311.17134](https://arxiv.org/abs/2311.17134)                                     |
| 2D spectra               | **2DNMRGym**                         | 22k+ HSQC spectra + SMILES (partly human‑annotated)                  | Train/benchmark 2D NMR predictors                            | [https://arxiv.org/abs/2505.18181](https://arxiv.org/abs/2505.18181)                                     |

---

#### 🛠️ Tools for Synthetic NMR Generation

| Tool                       | What it does                                                                       | How to use                                                            | Required Inputs                                                                                                        | Output                                              | Link                                                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Spinach (Matlab)**       | Physics‑based simulation of 1D/2D spectra (COSY, HSQC, NOESY), relaxation, MAS NMR | Define spin system + select pulse sequence + run simulation in Matlab | Spin system: isotopes, chemical shifts (ppm), J couplings (Hz), CSA/dipolar terms; external field (B₀); pulse sequence | Time‑domain FID & frequency‑domain spectra (1D/2D)  | [https://en.wikipedia.org/wiki/Spinach\_(software)](https://en.wikipedia.org/wiki/Spinach_%28software%29)        |
| **ORCA / NWChem**          | Quantum chemistry calculation of shielding tensors and J couplings                 | Run DFT/ab initio with NMR keyword                                    | 3D geometry (XYZ/MOL/PDB), basis set, method, optional solvent model                                                   | Isotropic shieldings (σ), J‑couplings, CSA tensors  | [https://github.com/nwchemgit/nwchem](https://github.com/nwchemgit/nwchem)                                       |
| **NMRDB**                  | Online prediction of ¹H/¹³C (1D and 2D COSY/HSQC/HMBC)                             | Draw molecule or paste SMILES on web UI                               | Molecular structure (drawn or SMILES), optional solvent/field strength                                                 | Simulated 1D/2D spectra, JCAMP export               | [https://www.nmrdb.org/](https://www.nmrdb.org/)                                                                 |
| **ChemAxon NMR Predictor** | Predicts ¹H/¹³C chemical shifts and spectra (GUI + CLI)                            | Use MarvinSketch or `cxcalc nmr` CLI                                  | Structure input (SMILES, MOL, SDF), optional solvent/field                                                             | Chemical shifts, spectra, JCAMP files               | [https://docs.chemaxon.com/display/docs/NMR%2BPredictor](https://docs.chemaxon.com/display/docs/NMR%2BPredictor) |
| **NMRbox**                 | VM platform bundling many NMR tools (TopSpin, Sparky, CCPN, etc.)                  | Launch VM, import spectra, run pipelines                              | Experimental or synthetic spectra, peak lists                                                                          | Processed spectra, assignments, structural analysis | [https://nmrbox.nmrhub.org/software](https://nmrbox.nmrhub.org/software)                                         |

---



## 4. IR Spectroscopy (Small Molecules)

### 4.1 Forward Task (Molecule → Spectrum)  
*Infrared spectrum prediction from molecular structures*  
### 📊 Forward Task Method Table (Molecule → Spectrum)

| Paper Title & Link | Published in | Method Type | Input | Metric | Data Source | Code | CKPT |
|------------|---------------|-------|-------|----------|--------------------|------|------|
| [Machine Learning Molecular Dynamics for the Simulation of Infrared Spectra (2017)](https://doi.org/10.1039/c7sc02267k) | Chem. Sci. | ML (HDNNP + NN Dipole) | 3D coordinates  | MAE | DFT | — | — |
| [A Machine Learning Protocol for Predicting Protein Infrared Spectra (2020)](https://pubs.acs.org/doi/10.1021/jacs.0c06530) | J. Am. Chem. Soc. | ML (MLP) | 3D coordinates  | RMSE + Spearman | DFT | — | — |
| [Predicting Infrared Spectra with Message Passing Neural Networks (2021)](https://doi.org/10.1021/acs.jcim.1c00055) | J. Chem. Inf. Model. | DL (MPNN + FFNN) |  **Comp:** SMILES<br>**Exp:** SMILES+Phase | SIS | **Comp:** PubChem<br>**Exp:** NIST, PNNL, AIST, Coblentz  | [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="24"/>](https://github.com/gfm-collab/chemprop-IR) | <a href="https://zenodo.org/records/4698943"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Graphormer-IR: Graph Transformers Predict Experimental IR Spectra (2024)](https://doi.org/10.1021/acs.jcim.4c00378) | J. Chem. Inf. Model. | DL (Graph Transformer + MLP, 1D-CNN) | SMILES+Phase | SIS | NIST, AIST, Coblentz | [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="24"/>](https://github.com/HopkinsLaboratory/Graphormer-IR) | <a href="https://zenodo.org/records/10790190"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |


### 4.2 Inverse Task (Spectrum → Molecule)  
*Molecular characterization from infrared spectra*  

---

## 5. Multimodal Spectroscopy (Small Molecules)
### 5.1 Forward Task (Molecule → Multiple Spectra)
*Joint prediction of multiple spectral modalities from molecular structures*

### 5.2 Inverse Task (Multiple Spectra → Molecule)
*Multimodal integration for enhanced molecular identification*

---

## 6. X-ray Diffraction (XRD) (Crystals)

### 6.1 Forward Task (Crystal → Pattern)  
*Prediction of XRD patterns from crystal structures*  
 

### 6.2 Inverse Task (Pattern → Crystal)  
*Crystal structure determination from XRD patterns*  


---

## License  
📄 This project is licensed under the MIT License — see the LICENSE file for details.

---
