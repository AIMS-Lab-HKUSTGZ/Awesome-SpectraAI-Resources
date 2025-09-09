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
| [Inferring CID by Estimating Breakage Events and Reconstructing their Graphs ](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.3c04654) | GNN, Transformer | Analytical Chemistry | [![Star](https://img.shields.io/github/stars/coleygroup/ms-pred.svg?style=social&label=Star)](https://github.com/coleygroup/ms-pred)  |ICEBERG|
| [Efficiently predicting high resolution mass spectra with graph neural networks](https://proceedings.mlr.press/v202/murphy23a.html) | GNN | ICML2023 | [![Star](https://img.shields.io/github/stars/murphy17/graff-ms.svg?style=social&label=Star)](https://github.com/murphy17/graff-ms)  |GRAFF-MS |
| [Tandem mass spectrum prediction for small molecules using graph transformers](https://doi.org/10.1038/s42256-024-00816-8) | Graph Transformer, Deep Learning Model | Nature Machine Intelligence | [![Star](https://img.shields.io/github/stars/Roestlab/massformer.svg?style=social&label=Star)](https://github.com/Roestlab/massformer) | MassFormer |
| [Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks](https://doi.org/10.1021/acscentsci.9b00085) | Neural Network, Graph-Convolutional Network | ACS Central Science | [![Star](https://img.shields.io/github/stars/brain-research/deep-molecular-massspec.svg?style=social&label=Star)](https://github.com/brain-research/deep-molecular-massspec) | NEIMS, CFM-EI |

### 1.2 Inverse Task (Spectrum → Molecule)  
*AI methods for molecular identification and elucidation from mass spectra*  
 | Paper Title & Link | Method Type | Venue | Code | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [An end-to-end deep learning framework for translating mass spectra to de-novo molecules](https://www.nature.com/articles/s42004-023-00932-3) |GRU, CNN |communications chemistry| [![Star](https://img.shields.io/github/stars/KavrakiLab/Spec2Mol.svg?style=social&label=Star)](https://github.com/KavrakiLab/Spec2Mol)  |Spec2mol |
| [Searching molecular structure databases with tandem mass spectra using CSI:FingerID](https://doi.org/10.1073/pnas.1509788112) | Machine Learning | Proceedings of the National Academy of Sciences | [project website](https://ww16.csi-fingerid.org/?sub1=20250830-1925-2033-949e-1db2a86ecef8) | CSI:FingerID |
| [Deep kernel learning improves molecular fingerprint prediction from tandem mass spectra](https://doi.org/10.1093/bioinformatics/btac260) | DNN, Kernel-based Model, SVM | Bioinformatics | [project website](https://bio.informatik.uni-jena.de/software/sirius) | Deep kernel learning method |
| [CSU-MS $^2$ : A Contrastive Learning Framework for Cross-Modal Compound Identification from MS/MS Spectra to Molecular Structures](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01594) | Contrastive Learning Framework, Transformer, GNN | Analytical Chemistry | [![Star](https://img.shields.io/github/stars/tingxiecsu/CSU-MS2.svg?style=social&label=Star)](https://github.com/tingxiecsu/CSU-MS2) | CSU-MS² |
| [MassGenie: A Transformer-Based Deep Learning Method for Identifying Small Molecules from Their Mass Spectra](https://www.mdpi.com/2218-273X/11/12/1793) | Transformer, Variational Autoencoder | Biomolecules  | [![Star](https://img.shields.io/github/stars/neilswainston/FragGenie.svg?style=social&label=Star)](https://github.com/neilswainston/FragGenie) | MassGenie, FragGenie, VAE-Sim |
| [JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data](https://doi.org/10.1093/bioinformatics/btaf354) | Joint Embedding Space Technique, GNN, MLP | Bioinformatics | [![Star](https://img.shields.io/github/stars/HassounLab/JESTR1.svg?style=social&label=Star)](https://github.com/HassounLab/JESTR1/) | JESTR |
| [Metabolite Identification through Machine Learning — Tackling CASMI Challenge Using FingerID](https://doi.org/10.3390/metabo3020484) | Machine Learning, Kernel-based approach, SVM | Metabolites |  | FingerID |
| [Annotating metabolite mass spectra with domain-inspired chemical formula transformers](https://www.nature.com/articles/s42256-023-00708-3) | Transformer, Neural Network, Contrastive Learning Model | Nature Machine Intelligence | [![Star](https://img.shields.io/github/stars/samgoldman97/mist.svg?style=social&label=Star)](https://github.com/samgoldman97/mist) | MIST |
| [Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS](https://doi.org/10.1038/s41587-025-02663-3) | Transformer, Foundation Model | Nature Biotechnology | [![Star](https://img.shields.io/github/stars/pluskal-lab/DreaMS.svg?style=social&label=Star)](https://github.com/pluskal-lab/DreaMS) | DreaMS, Foundation model |
| [Systematic classification of unknown metabolites using high-resolution fragmentation mass spectra](https://doi.org/10.1038/s41587-020-0740-8) | DNN, SVMs | Nature Biotechnology | [![Star](https://img.shields.io/github/stars/boecker-lab/sirius-libs.svg?style=social&label=Star)](https://github.com/boecker-lab/sirius-libs) | CANOPUS |
| [Automatic Compound Annotation from Mass Spectrometry Data Using MAGMa](https://doi.org/10.5702/massspectrometry.s0033) | Software, Substructure-based algorithm | Mass Spectrometry | [project website](www.emetabolomics.org/magma) | MAGMa |
| [DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra](http://arxiv.org/pdf/2502.09571v2) | Transformer, Discrete Graph Diffusion Model | arXiv | [![Star](https://img.shields.io/github/stars/coleygroup/DiffMS.svg?style=social&label=Star)](https://github.com/coleygroup/DiffMS) | DiffMS |
| [MetFID: artificial neural network-based compound fingerprint prediction for metabolite annotation](https://doi.org/10.1007/s11306-020-01726-7) | Artificial Neural Network | Metabolomics |  | MetFID |
| [MADGEN - MASS-SPEC ATTENDS TO DE NOVO MOLECULAR GENERATION](https://arxiv.org/abs/2501.01950) | Attention-based generative model, LSTM, CNN | arxiv| [![Star](https://img.shields.io/github/stars/HassounLab/MADGEN.svg?style=social&label=Star)](https://github.com/HassounLab/MADGEN) | MADGEN |
| [An end-to-end deep learning framework for translating mass spectra to de-novo molecules](https://doi.org/10.1038/s42004-023-00932-3) | Deep Learning architecture, Autoencoder | Communications Chemistry | [![Star](https://img.shields.io/github/stars/KavrakiLab/Spec2Mol.svg?style=social&label=Star)](https://github.com/KavrakiLab/Spec2Mol) | Spec2Mol |
| [DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models](http://arxiv.org/pdf/2507.06853v1) | Diffusion Model, Transformer | arXiv |  | DiffSpectra |
| [MS2Mol: A transformer model for illuminating dark chemical space from mass spectra](https://doi.org/10.26434/chemrxiv-2023-vsmpx-v3) | Transformer, Generative Model |  |  | MS2Mol |
| [Improved metabolite identification with MIDAS and MAGMa through MS/MS spectral dataset-driven parameter optimization](https://doi.org/10.1007/s11306-016-1036-3) | Machine Learning, Rule-based | Metabolomics | [![Star](https://img.shields.io/github/stars/savantas/MAGMa-plus.svg?style=social&label=Star)](https://github.com/savantas/MAGMa-plus) | MIDAS, MAGMa, CSI: FingerID |
| [Using Graph Neural Networks for Mass Spectrometry Prediction](http://arxiv.org/pdf/2010.04661v1) | GNN | arXiv |  | GNN-based models |

### 1.3 General Tools
 | Paper Title & Link | Feasible scene | Venue | Code | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [matchms- processing and similarity evaluation of mass spectrometry data](https://joss.theoj.org/papers/10.21105/joss.02411) |raw mass spectra to pre- and post-processe  | The Journal of Open Source Software |[![Star](https://img.shields.io/github/stars/matchms/matchms.svg?style=social&label=Star)](https://github.com/matchms/matchms)  |python package|
| [MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4) | compare tandem mass spectra | Journal of Cheminformatics| [![Star](https://img.shields.io/github/stars/matchms/ms2deepscore.svg?style=social&label=Star)](https://github.com/matchms/ms2deepscore) | |
| [Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships](https://doi.org/10.1371/journal.pcbi.1008724) | NLP-inspired Model, Word2Vec | PLOS Computational Biology  | [![Star](https://img.shields.io/github/stars/iomega/spec2vec.svg?style=social&label=Star)](https://github.com/iomega/spec2vec) | Spec2Vec |
| [Chemically informed analyses of metabolomics mass spectrometry data with Qemistree](https://www.nature.com/articles/s41589-020-00677-3) | Machine Learning, Tree-based approach | Nature chemical biology | [![Star](https://img.shields.io/github/stars/biocore/q2-qemistree.svg?style=social&label=Star)](https://github.com/biocore/q2-qemistree) | Qemistree |
| [MSNovelist: de novo structure generation from mass spectra](https://doi.org/10.1038/s41592-022-01486-3) | Encoder-decoder neural network, RNN | Nature Methods | [![Star](https://img.shields.io/github/stars/meowcat/MSNovelist.svg?style=social&label=Star)](https://github.com/meowcat/MSNovelist) | MSNovelist |

### 1.4 Datasets, Benchmark and Review
 | Name | Type | Size | Website | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [MassSpecGym: A benchmark for the discovery and identification of molecules](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html) | Benchmark, Transformer, GNN | Advances in Neural Information Processing Systems | [![Star](https://img.shields.io/github/stars/pluskal-lab/MassSpecGym.svg?style=social&label=Star)](https://github.com/pluskal-lab/MassSpecGym) | MassSpecGym |
| [Artificial Intelligence in Spectroscopy: Advancing Chemistry from Prediction to Generation and Beyond](http://arxiv.org/pdf/2502.09897v1) | Neural architectures, ML-empowered solution | arXiv |  | review |


---

## 2. Mass Spectrometry (Peptides)

### 2.1 Forward Task (Peptides → Spectrum)  
*Computational methods for predicting peptides mass spectra*  


### 2.2 Inverse Task (Spectrum → Peptides)  
*AI approaches for peptides identification and quantification*  


---

## 3. NMR Spectroscopy (Small Molecules)

### 3.1 Forward Task (Molecule → Spectrum)

| Paper Title & Link                                                                                                                                         | Method Type                    | Venue/Year      | Data Source             | Metric       | Code / Data                                                                                                                                     | Notes                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | --------------- | ----------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [Towards a Unified Benchmark and Framework for Deep Learning-Based Prediction of NMR Chemical Shifts](https://arxiv.org/pdf/2408.15681) | Foundation pretrain + finetune | arXiv 2023      | 3D molecular structures | MAE (ppm)    | [NMRNet](https://github.com/Colin-Jay/NMRNet) · [Zenodo](https://zenodo.org/records/13317524)                                                   | Masked pretraining on 3D; unified benchmark framing |
| [PROSPRE: Solvent-aware ^1H NMR chemical shift prediction using deep learning](https://www.mdpi.com/2218-1989/14/5/290) | Deep Learning + Solvent-aware   | Metabolites 2024 | Experimental NMR data   | MAE < 0.10 ppm | —                                                                                                                                               | Trained on large-scale solvent-specific datasets    |
[Prediction of chemical shift in NMR: A review](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/mrc.5234)                                                                                                              | Review                         | —               | —                       | —            | —                                                                                                                                               | Survey of methods & datasets                        |
| [iShiftML: Highly Accurate Prediction of NMR Chemical Shifts](https://doi.org/10.1002/pca.3292)                                                                                                               | Hybrid ML + QM                 | —               | QM descriptors          | < 0.2 ppm    | —                                                                                                                                               | Fast inference; QM features required                |
| [TransPeakNet: Solvent-Aware 2D NMR Prediction via Multi-Task Pre-Training and Unsupervised Learning](https://github.com/siriusxiao62/2dNMR)      | GNN + Multitask                | Chem paper 2023 | Graph-based input       | 2D spectra   | [Code](https://github.com/siriusxiao62/2dNMR) · [Data](https://drive.google.com/drive/folders/1wQxk7mnIwi5aAGaF34_hk7xo6IeEh-IE?usp=drive_link) | 1D→2D solvent-aware prediction                      |
| [NMR shift prediction from small data quantities](https://link.springer.com/article/10.1186/s13321-023-00785-x)                                                                                                            | ML                             | —               | NMRShiftDB2             | MAE (ppm)    | —                                                                                                                                               | Small-data learning                                 |
| [NMR-spectrum prediction for dynamic molecules](https://pubs.aip.org/aip/jcp/article/158/19/194108/2891394)                                                                                                              | ML-Dynamics                    | —               | Simulated ensembles     | Time-avg ppm | —                                                                                                                                               | Conformational averaging                            |
| [Machine learning in NMR spectroscopy](https://www.sciencedirect.com/science/article/pii/S0079656525000196)                                                                                                                       | Review (DL)                    | —               | NMRShiftDB2             | —            | —                                                                                                                                               | Multitask trends & outlook                          |
| [A framework for automated structure elucidation from routine NMR spectra](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc04105c)                                                                                   | ML-based structure elucidation | Chem. Sci. 2021 | 1D ^1H/^13C NMR spectra | Top-10 accuracy | —                                                                                                                                               | ML framework for automated structure elucidation    |
| [Deep learning enabled ultra-high quality NMR chemical shift prediction from spin echo spectra](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc04742g) | Deep Learning + Signal Processing | Science 2024    | Spin echo NMR spectra   | MAE (ppm)    | —                                                                                                                                               | High-resolution chemical shift prediction          |


---

### 3.2 Inverse Task (Spectrum → Molecule)

| Paper Title & Link                                                                                                                                                                                                                                                                                                           | Method Type                     | Venue/Year                | Input                  | Metric             | Code / Data | Notes                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------- | ---------------------- | ------------------ | ----------- | ----------------------------------------------------------------------- |
| [A Transformer Based Generative Chemical Language AI Model for Structural Elucidation of Organic Compounds](https://arxiv.org/abs/2410.14719) | Transformer-based Generative AI | Arxiv 2024                | IR, UV, ^1H NMR spectra | Top-15 accuracy      | —           | End-to-end structure elucidation via chemical language modeling         |
| [NMR-Solver: Automated Structure Elucidation via Large-Scale Spectral Matching and Physics-Guided Fragment Optimization](https://arxiv.org/abs/2509.00640) | Hybrid Spectral Matching + Physics-guided Optimization | Arxiv 2025                | ^1H/^13C NMR spectra    | Top-1 accuracy      | —           | Combines spectral matching with physics-guided optimization             |
| [Accurate and Efficient Structure Elucidation from Routine One-Dimensional NMR Spectra Using Multitask Machine Learning](https://arxiv.org/pdf/2408.08284)                                                                                                                                                | CNN + Transformer (multitask)   | Arxiv 2024                | 1D spectra             | Top-1 / Top-k      | —           | High accuracy with 1D only                                              |
| [Learning the Language of NMR Structure Elucidation from NMR Spectra Using Transformer Models](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/64d5e4ccdfabaf06ff1763ef/original/learning-the-language-of-nmr-structure-elucidation-from-nmr-spectra-using-transformer-models.pdf) | Transformer (sequence modeling) | ChemRxiv 2023             | 1D spectra             | —                  | —           | Treats NMR as a language; structure reasoning                           |
| [DeepSAT: Learning Molecular Structures from NMR Data](https://link.springer.com/article/10.1186/s13321-023-00738-4)                                                                                                                                                                                                                                                                       | Multimodal DL                   | —                         | NMR spectra            | Structure accuracy | —           | Uses NPAtlas, NPASS, GNPS etc.; multimodal molecular structure learning |
| [NMR Foundation Model for Structure Prediction](https://neurips.cc/virtual/2024/poster/97441)                                                                                                                                                                                                   | Transformer-based FM            | NeurIPS 2024              | 1D/2D NMR              | Accuracy (Top-k)   | —           | Early spectra foundation model in AI venue                              |
| [Bayesian approach to structural elucidation with crystalline-state SSNMR](https://arxiv.org/abs/1909.00870)                                                                                                                                                                                              | Bayesian / probabilistic        | arXiv 2019                | Solid-state NMR        | Top-5              | —           | Requires crystal info                                                   |
| [Deep RL + GCN for NMR inverse problem](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c00624)                                                                                                                                                                                                    | RL (MCTS) + GCN                 | J. Phys. Chem. Lett. 2022 | Shift table            | Top-3              | —           | Effective for small molecules                                           |

### 3.3 NMR Datasets & Benchmarks

| Name & Link                                                                                                                                                                          | Type                       | Venue/Year            | Size / Modality                | Real / Sim          | Download                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | --------------------- | ------------------------------ | ------------------- | ------------------------------------------------------ |
| [Multimodal Spectroscopic Dataset (includes NMR)](https://proceedings.neurips.cc/paper_files/paper/2024/file/e38e60b33bb2c6993e0865160cdb5cf1-Paper-Datasets_and_Benchmarks_Track.p) | Benchmark (multi-spectra)  | NeurIPS 2024          | 7.9e5+ synthetic spectra       | Sim                 | Zenodo (via paper)                                     |
| [IR–NMR Multimodal Computational Spectra Dataset](https://www.nature.com/articles/s41597-025-05729-8)                                                                                | Dataset (IR + NMR)         | Nature Sci. Data 2025 | 177,461 spectra                | Sim (MD + DFT + ML) | [Zenodo](https://zenodo.org/records/16417648)          |
| [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/)                                                                                                                                  | Database (small molecules) | —                     | \~50k; ¹H/¹³C                  | Real                | Open                                                   |
| [BMRB](https://bmrb.io/)                                                                                                                                                             | Database (bio-molecules)   | —                     | >13k biomolecules; ¹H/¹³C/¹⁵N… | Real                | FTP/STAR                                               |
| [SDBS](https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_index.cgi)                                                                                                                        | Database (multi-modal)     | —                     | \~14k; ¹H/¹³C/IR/MS/UV         | Real                | Crawl/script                                           |
| [2DNMRGym (HSQC)](https://arxiv.org/abs/2405.18181)                                                                                                                                  | Simulated 2D dataset       | 2024                  | 22k+ HSQC                      | Sim                 | [HuggingFace/Zenodo](https://huggingface.co/datasets/) |
| [NMRMixDB](https://nmrmixdb.github.io/)                                                                                                                                              | Mixtures                   | —                     | \~3k; ¹H                       | Real                | Open                                                   |


---

### 3.4 Clustering & Representation Learning for NMR

| Paper Title & Link                                                                                                     | Method                      | Venue/Year | Task                       | Notes                        |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------- | ---------- | -------------------------- | ---------------------------- |
| Statistical HOmogeneous Cluster SpectroscopY (SHOCSY) ([Anal. Chem. 2014](https://pubs.acs.org/doi/10.1021/ac500161k)) | Statistical clustering      | 2014       | ¹H metabolomics clustering | Classic baseline             |
| Sparse Convex Wavelet Clustering (includes NMR signals)                                                                | Wavelet + convex clustering | arXiv      | Signal clustering          | Joint denoising + clustering |
| Deep representation learning for NMR spectral clustering                                                               | Autoencoder / DL            | —          | Spectral clustering        | DL embedding → K-means/HC    |

---

### 3.5 Foundation Models & Chemistry LMs related to NMR

| Model / Paper                | Family                   | Venue/Year         | Link                                            | Notes                                                           |
| ---------------------------- | ------------------------ | ------------------ | ----------------------------------------------- | --------------------------------------------------------------- |
| ChemGPT                      | Large chemistry LM       | —                  | —                                               | Molecular generative & reasoning; spectra-to-structure transfer |
| DETANet                      | DL architecture          | —                  | —                                               | Chemistry perception; potential for spectra-conditioned tasks   |
| DreaMS (MS foundation model) | Transformer (spectra FM) | Nat. Biotech. 2025 | [GitHub](https://github.com/pluskal-lab/DreaMS) | Cross-modal FM idea transferrable to NMR                        |



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
| **MestReNova**       | Comprehensive NMR data analysis software for processing 1D/2D spectra, including peak picking, integration, and chemical shift assignment | Process NMR spectra, analyze data, and assign chemical shifts using MestReNova interface | NMR spectra: 1D, 2D, 3D; chemical shift (ppm), coupling constants (Hz), integration, peak assignment | Processed spectra, chemical shift assignments, peak integration | [MestReNova Official Website](https://mestrelab.com/software/mestreNova/) |
| **Spinach (Matlab)**       | Physics‑based simulation of 1D/2D spectra (COSY, HSQC, NOESY), relaxation, MAS NMR | Define spin system + select pulse sequence + run simulation in Matlab | Spin system: isotopes, chemical shifts (ppm), J couplings (Hz), CSA/dipolar terms; external field (B₀); pulse sequence | Time‑domain FID & frequency‑domain spectra (1D/2D)  | [https://en.wikipedia.org/wiki/Spinach\_(software)](https://en.wikipedia.org/wiki/Spinach_%28software%29)        |
| **ORCA / NWChem**          | Quantum chemistry calculation of shielding tensors and J couplings                 | Run DFT/ab initio with NMR keyword                                    | 3D geometry (XYZ/MOL/PDB), basis set, method, optional solvent model                                                   | Isotropic shieldings (σ), J‑couplings, CSA tensors  | [https://github.com/nwchemgit/nwchem](https://github.com/nwchemgit/nwchem)                                       |
| **NMRDB**                  | Online prediction of ¹H/¹³C (1D and 2D COSY/HSQC/HMBC)                             | Draw molecule or paste SMILES on web UI                               | Molecular structure (drawn or SMILES), optional solvent/field strength                                                 | Simulated 1D/2D spectra, JCAMP export               | [https://www.nmrdb.org/](https://www.nmrdb.org/)                                                                 |
| **ChemAxon NMR Predictor** | Predicts ¹H/¹³C chemical shifts and spectra (GUI + CLI)                            | Use MarvinSketch or `cxcalc nmr` CLI                                  | Structure input (SMILES, MOL, SDF), optional solvent/field                                                             | Chemical shifts, spectra, JCAMP files               | [https://docs.chemaxon.com/display/docs/NMR%2BPredictor](https://docs.chemaxon.com/display/docs/NMR%2BPredictor) |
| **NMRbox**                 | VM platform bundling many NMR tools (TopSpin, Sparky, CCPN, etc.)                  | Launch VM, import spectra, run pipelines                              | Experimental or synthetic spectra, peak lists                                                                          | Processed spectra, assignments, structural analysis | [https://nmrbox.nmrhub.org/software](https://nmrbox.nmrhub.org/software)                                         |

---



## 4. IR Spectroscopy (Small Molecules)

### 4.1 Forward Task (Molecule → Spectrum)  
*Infrared spectrum prediction from molecular structures*  
| Paper Title & Link | Venue | Method Type | Input |  Data Source | Code | CKPT |
|------------|---------------|-------|-------|--------------------|------|------|
| [Machine Learning Molecular Dynamics for the Simulation of Infrared Spectra (2017)](https://doi.org/10.1039/c7sc02267k) | Chem. Sci. | ML (HDNNP + NN Dipole) | 3D coordinates  | — | — | — |
| [A Machine Learning Protocol for Predicting Protein Infrared Spectra (2020)](https://pubs.acs.org/doi/10.1021/jacs.0c06530) | J. Am. Chem. Soc. | ML (MLP) | 3D coordinates  | — | — | — |
| [Predicting Infrared Spectra with Message Passing Neural Networks (2021)](https://doi.org/10.1021/acs.jcim.1c00055) | J. Chem. Inf. Model. | DL (MPNN + FFNN) |  **Simulated:** SMILES<br>**Experimental:** SMILES+Phase | **Simulated:** PubChem<br>**Experimental:** NIST, PNNL, AIST, Coblentz  | [![Star](https://img.shields.io/github/stars/gfm-collab/chemprop-IR.svg?style=social&label=Star)](https://github.com/gfm-collab/chemprop-IR) | <a href="https://zenodo.org/records/4698943"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Graphormer-IR: Graph Transformers Predict Experimental IR Spectra (2024)](https://doi.org/10.1021/acs.jcim.4c00378) | J. Chem. Inf. Model. | DL (Graph Transformer + MLP, 1D-CNN) | SMILES+Phase | NIST, AIST, Coblentz | [![Star](https://img.shields.io/github/stars/HopkinsLaboratory/Graphormer-IR.svg?style=social&label=Star)](https://github.com/HopkinsLaboratory/Graphormer-IR) | <a href="https://zenodo.org/records/10790190"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Neural Network Approach for Predicting Infrared Spectra from 3D Molecular Structure (2024)](https://doi.org/10.1016/j.cplett.2024.141603) | Chem. Phys. Lett. | DL (MPNN) | 3D Molecular Structure | NIST | [![Star](https://img.shields.io/github/stars/allouchear/NNMol-IR.svg?style=social&label=Star)](https://github.com/allouchear/NNMol-IR) | <a href="https://zenodo.org/records/13681778"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Infrared Spectra Prediction for Functional Group Region Utilizing a Machine Learning Approach with Structural Neighboring Mechanism (2024)](https://doi.org/10.1021/acs.analchem.4c01972) | Anal. Chem. | ML | SMILES | CIAD | [![Star](https://img.shields.io/github/stars/ChengchunLiu/IR-DIAZO-KETONE.svg?style=social&label=Star)](https://github.com/ChengchunLiu/IR-DIAZO-KETONE) | -- |
| [Infrared Spectra Prediction Using Attention-Based Graph Neural Networks (2024)](https://doi.org/10.1039/d3dd00254c) | Digital Discovery | DL (GNN) | SMILES | NIST | [![Star](https://img.shields.io/github/stars/nj-saquer/IR-Spectra-Prediction-Graph-Models.svg?style=social&label=Star)](https://github.com/nj-saquer/IR-Spectra-Prediction-Graph-Models) | — |
| [Neural Network Approach for Predicting Infrared Spectra from 3D Molecular Structure (2024)](https://arxiv.org/abs/2405.05737) | arXiv | DL (MPNN) | 3D Molecular Structure | NIST | [![Star](https://img.shields.io/github/stars/allouchear/NNMol-IR.svg?style=social&label=Star)](https://github.com/allouchear/NNMol-IR) | <a href="https://zenodo.org/records/13681778"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Prediction of the Infrared Absorbance Intensities and Frequencies of Hydrocarbons: A Message Passing Neural Network Approach (2024)](https://doi.org/10.1021/acs.jpca.4c06745) | J. Phys. Chem. A | DL (MPNN + FFNN) | SMILES | GDB | <a href="https://zenodo.org/records/13844305"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | <a href="https://zenodo.org/records/13844305"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Unlocking the Potential of Machine Learning in Enhancing Quantum Chemical Calculations for Infrared Spectral Prediction (2025)](https://doi.org/10.1021/acsomega.5c02405) | ACS Omega | ML (Multioutput Regressor + Random Forest Regressor) | 3D Molecular Structure | Gaussian 16 | Supporting Information | — |



### 4.2 Inverse Task (Spectrum → Molecule)  
*Molecular characterization from infrared spectra*  
| Paper Title & Link | Venue | Method Type | Output | Data Source | Code | CKPT |
|------------|---------------|-------|-------|--------------------|------|------|
| [Can LLMs Solve Molecule Puzzles? A Multimodal Benchmark for Molecular Structure Elucidation (2024)](https://kehanguo2.github.io/Molpuzzle.io/) | NeurIPS2024 | LLM (GPT‑4o, Claude‑3, Gemini, etc.) | SMILES | — | [![Star](https://img.shields.io/github/stars/KehanGuo2/MolPuzzle.svg?style=social&label=Star)](https://github.com/KehanGuo2/MolPuzzle) | — |
| [Leveraging Infrared Spectroscopy for Automated Structure Elucidation (2024)](https://doi.org/10.1038/s42004-024-01341-w) | Commun. Chem. | DL (Transformer) | SMILES | **Simulated:** PubChem<br>**Experimental:** NIST | [![Star](https://img.shields.io/github/stars/rxn4chemistry/rxn-ir-to-structure.svg?style=social&label=Star)](https://github.com/rxn4chemistry/rxn-ir-to-structure) | <a href="https://zenodo.org/records/7928396"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Transformer-Based Models for Predicting Molecular Structures from Infrared Spectra Using Patch-Based Self-Attention (2025)](https://doi.org/10.1021/acs.jpca.4c05665) | J. Phys. Chem. A | DL (Transformer + Patch-based Self-Attention) | SMILES | **Simulated:** PubChem, QM9S <br>**Experimental:** NIST  | [![Star](https://img.shields.io/github/stars/wenjin886/PatchBasedSelfAttention.svg?style=social&label=Star)](https://github.com/wenjin886/PatchBasedSelfAttention) | <a href="https://zenodo.org/records/12789777"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> |
| [Revolutionizing Spectroscopic Analysis Using Sequence-to-Sequence Models I: From Infrared Spectra to Molecular Structures (2025)](https://doi.org/10.26434/chemrxiv-2025-n4q84) | ChemRxiv | DL (GRU, LSTM, GPT, Transformer) | SELFIES | QM9, PC9 | — | — |

### 4.3 IR Datasets
| Dataset / Method Name & Link | Size | Data Source | Real / Simulated | Element Coverage | 
|---------------------|----------------|------------------|-----|----------------------|
| [Chemprop-IR](https://doi.org/10.1021/acs.jcim.1c00055) <a href="https://zenodo.org/records/4698943"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | 85,232 | PubChem (SMILES for molecular structures) | Simulated (GFN2-xTB) | C, H, O, N, S, P, Si, F, Cl, Br, I |
| [CMPNN](https://doi.org/10.1021/acs.jpca.4c06745) <a href="https://zenodo.org/records/13844305"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | 31,570 | GDB (SMILES for molecular structures) | Simulated (DFT) | C, H |
| [Multimodal Spectroscopic Dataset](https://proceedings.neurips.cc/paper_files/paper/2024/file/e38e60b33bb2c6993e0865160cdb5cf1-Paper-Datasets_and_Benchmarks_Track.pdf) <a href="https://zenodo.org/records/11611178"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | 794,403 | USPTO reaction dataset (SMILES for molecular structures) | Simulated (MD) | C, H, O, N, S, P, Si, B, F, Cl, Br, I |
| [IRtoMol](https://doi.org/10.1038/s42004-024-01341-w) <a href="https://zenodo.org/records/7928396"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | 634,585 | PubChem (SMILES for molecular structures) | Simulated (MD + PCFF forcefield) | C, H, O, N, S, P, F, Cl, Br, I |
| [MolPuzzle](https://proceedings.neurips.cc/paper_files/paper/2024/file/f2b9e8e7a36d43ddfd3d55113d56b1e0-Paper-Datasets_and_Benchmarks_Track.pdf) <a href="https://github.com/KehanGuo2/MolPuzzle"><img src="https://api.iconify.design/mdi:github.svg" width="20"/></a> | 216 (Picture format) | — | Mixed | — |
| [IR–NMR Multimodal Computational Spectra Dataset](https://www.nature.com/articles/s41597-025-05729-8) <a href="https://zenodo.org/records/16417648"><img src="https://api.iconify.design/academicons:zenodo.svg" width="20"/></a> | 177,461 | USPTO (SMILES for molecular structures) | Simulated (MD + DFT + ML) | C, H, O, N, S, P, Si, B, F, Cl, Br, I |
| [QM9S](https://www.nature.com/articles/s43588-023-00550-y) <a href="https://figshare.com/articles/dataset/QM9S_dataset/24235333"><img src="https://api.iconify.design/academicons:figshare.svg" width="20"/></a> | 133,885 | QM9 (re-optimized geometries) | Simulated (DFT) | C, H, O, N, F  |
| SRD 35 <a href="https://www.nist.gov/srd/nist-standard-reference-database-35"><img src="https://cdn.worldvectorlogo.com/logos/nist.svg" width="20"/></a> | 5,228 (gas-phase) | — | Real | — |
| NIST Chemistry WebBook <a href="https://webbook.nist.gov/chemistry/"><img src="https://cdn.worldvectorlogo.com/logos/nist.svg" width="20"/></a> | >16,000 | — | Real | — |
| Coblentz Society Spectral Database <a href="https://webbook.nist.gov/chemistry/coblentz/"><img src="https://cdn.worldvectorlogo.com/logos/nist.svg" width="20"/></a> | >9,500 | — | Real | — |
| [NWIR](https://www.pnnl.gov/publications/northwest-infrared-nwir-gas-phase-spectral-database-industrial-and-environmental) | ~1,000–1,500 (gas-phase) | — | Real | — |
| [AIST SDBS](https://sdbs.db.aist.go.jp) | ~54,100 | — | Real | — |





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
