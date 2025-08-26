# Awesome-SpectraAI-Resources
<div align="center">
  <img width="690" height="83" alt="Awesome SpectraAI Resources" src="https://github.com/user-attachments/assets/5ecb4f23-cb12-46c8-bd8b-5f13481a5850" />
</div>

✨✨ A curated collection of resources on artificial intelligence for spectral data analysis, covering computational methods for mass spectrometry (MS), NMR, IR, and XRD data.

---

## 1. Mass Spectrometry (Small Molecules)

### 1.1 Forward Task (Molecule → Spectrum)  
*Computational approaches for predicting mass spectra from molecular structures*  
 
 | Paper Title & Link | Method Type | Venue | Code | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [Efficiently predicting high resolution mass spectra with graph neural networks](https://proceedings.mlr.press/v202/murphy23a.html) | GNN | ICML2023 | [![Star](https://img.shields.io/github/stars/murphy17/graff-ms.svg?style=social&label=Star)](https://github.com/murphy17/graff-ms)  | low consumption, fast|




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

## 4. IR Spectroscopy (Small Molecules)

### 4.1 Forward Task (Molecule → Spectrum)  
*Infrared spectrum prediction from molecular structures*  


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
