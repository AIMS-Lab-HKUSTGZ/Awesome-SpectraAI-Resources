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
| [Inferring CID by Estimating Breakage Events and Reconstructing their Graphs ](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.3c04654) | GNN, Transformer | analytical chemistry | [![Star](https://img.shields.io/github/stars/coleygroup/ms-pred.svg?style=social&label=Star)](https://github.com/coleygroup/ms-pred)  |ICEBERG|
| [Efficiently predicting high resolution mass spectra with graph neural networks](https://proceedings.mlr.press/v202/murphy23a.html) | GNN | ICML2023 | [![Star](https://img.shields.io/github/stars/murphy17/graff-ms.svg?style=social&label=Star)](https://github.com/murphy17/graff-ms)  |GRAFF-MS |
| [CFM-ID 4.0: More Accurate ESI MS/MS Spectral Prediction and Compound Identification](https://pubs.acs.org/doi/pdf/10.1021/acs.analchem.1c01465?ref=article_openPDF) | Machine Learning Model | analytical chemistry |  | CFM-ID 4.0, [project website](http://cfmid4.wishartlab.com/) |
| [Computational prediction of electron ionization mass spectra to assist in GC-MS compound identification](https://pubs.acs.org/doi/pdf/10.1021/acs.analchem.6b01622?ref=article_openPDF) | Artificial Neural Network, Probabilistic Generative Model | analytical chemistry | | CFM-EI,old version of CFM-ID, [project website](http://cfmid4.wishartlab.com/) |
| [Tandem mass spectrum prediction for small molecules using graph transformers](https://doi.org/10.1038/s42256-024-00816-8) | Graph Transformer, Deep Learning Model | Nature Machine Intelligence | [![Star](https://img.shields.io/github/stars/Roestlab/massformer.svg?style=social&label=Star)](https://github.com/Roestlab/massformer) | MassFormer |
| [Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks](https://doi.org/10.1021/acscentsci.9b00085) | Neural Network, Graph-Convolutional Network | ACS Central Science | [![Star](https://img.shields.io/github/stars/brain-research/deep-molecular-massspec.svg?style=social&label=Star)](https://github.com/brain-research/deep-molecular-massspec) | NEIMS |
| [Mass Spectra Prediction with Structural Motif-based Graph Neural Networks](https://doi.org/10.1038/s41598-024-51760-x) | GNN, MLP, Graph Transformer | Scientific Reports |  | MoMS-Net |
| [Prediction of electron ionization mass spectra based on graph convolutional networks](https://doi.org/10.1016/j.ijms.2022.116817) | GCN, MLP | International Journal of Mass Spectrometry |  | Baojie Zhang mass spectra prediction |
|[QCxMS and QCEIMS related publications](https://xtb-docs.readthedocs.io/en/latest/qcxms_doc/qcxms_cites.html)|Born-Oppenheimer Molecular Dynamics|| [![Star](https://img.shields.io/github/stars/qcxms/QCxMS.svg?style=social&label=Star)](https://github.com/qcxms/QCxMS) |QCxMS, QCxMS2, QCEIMS, First Principles Calculation, [project website](https://xtb-docs.readthedocs.io/en/latest/qcxms_doc/qcxms_cites.html)|





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
| [MSNovelist: de novo structure generation from mass spectra](https://doi.org/10.1038/s41592-022-01486-3) | Encoder-decoder neural network, RNN | Nature Methods | [![Star](https://img.shields.io/github/stars/meowcat/MSNovelist.svg?style=social&label=Star)](https://github.com/meowcat/MSNovelist) | MSNovelist |
| [Predicting a Molecular Fingerprint from an Electron Ionization Mass Spectrum with Deep Neural Networks](https://pubs.acs.org/doi/10.1021/acs.analchem.0c01450) | Deep Neural Network | analytical chemistry | [![Star](https://img.shields.io/github/stars/hcji/DeepEI.svg?style=social&label=Star)](https://github.com/hcji/DeepEI) | DeepEI, FP Model, molecular fingerprint prediction |
| [Deep MS/MS-Aided Structural-Similarity Scoring for Unknown Metabolite Identification](https://doi.org/10.1021/acs.analchem.8b05405.s004) | Deep Neural Network |  | [![Star](https://img.shields.io/github/stars/hcji/DeepMASS.svg?style=social&label=Star)](https://github.com/hcji/DeepMASS) | DeepMASS, mass spectrum-based library match for Metabolite  |
| [Ultra-fast and accurate electron ionization mass spectrum matching for compound identification with million-scale in-silico library](https://www.nature.com/articles/s41467-023-39279-7)|Word2vec, HNSW|nature communications|[![Star](https://img.shields.io/github/stars/Qiong-Yang/FastEI.svg?style=social&label=Star)](https://github.com/Qiong-Yang/FastEI)|FastEI, spectrum simulation expansion, spectrum search for compoundidentification|
| [In silico fragmentation for computer assisted identification of metabolite mass spectra](https://doi.org/10.1186/1471-2105-11-148) | Combinatorial Fragmenter | BMC Bioinformatics |  | MetFrag, mass spectrum-based metabolite identification, [Project Link](http://msbi.ipb-halle.de/MetFrag/) |
| [MS2Query: reliable and scalable MS2 mass spectra-based analogue search](https://doi.org/10.1038/s41467-023-37446-4) |embedding-based chemical similarity predictors  | Nature Communications | [![Star](https://img.shields.io/github/stars/iomega/ms2query.svg?style=social&label=Star)](https://github.com/iomega/ms2query) | MS2Query, mass spectra analogue search, Project Link: [https://doi.org/10.5281/zenodo.6124553](https://doi.org/10.5281/zenodo.6124553) |

### 1.3 General Tools
 | Paper Title & Link | Feasible scene | Venue | Code | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [matchms- processing and similarity evaluation of mass spectrometry data](https://joss.theoj.org/papers/10.21105/joss.02411) |raw mass spectra to pre- and post-processe  | The Journal of Open Source Software |[![Star](https://img.shields.io/github/stars/matchms/matchms.svg?style=social&label=Star)](https://github.com/matchms/matchms)  |python package|
| [MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4) | compare tandem mass spectra | Journal of Cheminformatics| [![Star](https://img.shields.io/github/stars/matchms/ms2deepscore.svg?style=social&label=Star)](https://github.com/matchms/ms2deepscore) | |
| [Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships](https://doi.org/10.1371/journal.pcbi.1008724) | NLP-inspired Model, Word2Vec | PLOS Computational Biology  | [![Star](https://img.shields.io/github/stars/iomega/spec2vec.svg?style=social&label=Star)](https://github.com/iomega/spec2vec) | Spec2Vec |
| [Chemically informed analyses of metabolomics mass spectrometry data with Qemistree](https://www.nature.com/articles/s41589-020-00677-3) | Machine Learning, Tree-based approach | Nature chemical biology | [![Star](https://img.shields.io/github/stars/biocore/q2-qemistree.svg?style=social&label=Star)](https://github.com/biocore/q2-qemistree) | Qemistree |
| [MetaboAnalystR 4.0: a unified LC-MS workflow for global metabolomics](https://doi.org/10.1038/s41467-024-48009-6) | Software Tool | Nature Communications | [![Star](https://img.shields.io/github/stars/xia-lab/MetaboAnalystR.svg?style=social&label=Star)](https://github.com/xia-lab/MetaboAnalystR) | MetaboAnalystR 4.0, LC-MS data processing, R package, Project Link: [MetaboAnalyst](https://www.metaboanalyst.ca/docs/Databases.xhtml) |
| [Fully Automated Unconstrained Analysis of High-Resolution Mass Spectrometry Data with Machine Learning](https://doi.org/10.1021/jacs.2c03631) | Decision Trees, Neural Network, LSTM | Journal of the American Chemical Society | [![Star](https://img.shields.io/github/stars/Ananikov-Lab/medusa.svg?style=social&label=Star)](https://github.com/Ananikov-Lab/medusa) | MEDUSA, mass spectrum analysis overall framework, Project Link: [project_link](https://ananikov-lab.github.io/medusa/) |
| [The METLIN small molecule dataset for machine learning-based retention time prediction](https://doi.org/10.1038/s41467-019-13680-7) | Deep Learning | Nature Communications |  | [project website](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913), retention time prediction, R package |
| [An end-to-end deep learning method for mass spectrometry data analysis to reveal disease-specific metabolic profiles](https://doi.org/10.1038/s41467-024-51433-3) | Deep Learning | Nature Communications | [![Star](https://img.shields.io/github/stars/yjdeng9/DeepMSProfiler.svg?style=social&label=Star)](https://github.com/yjdeng9/DeepMSProfiler) | DeepMSProfiler, disease-specific metabolic profiling |
| [Trackable and scalable LC-MS metabolomics data processing using asari](https://doi.org/10.1038/s41467-023-39889-1) | open-source software tool | Nature Communications | [![Star](https://img.shields.io/github/stars/shuzhao-li/asari.svg?style=social&label=Star)](https://github.com/shuzhao-li/asari) | asari, LC-MS data processing, Project Link: [https://pypi.org/project/asari-metabolomics/](https://pypi.org/project/asari-metabolomics/) |

### 1.4 Datasets, Benchmark and Review
 | Name | Type | Size | Website | Notes |
|--------------------|------------------|--------------------|--------------------|------------------------------------------|
| [MassSpecGym: A benchmark for the discovery and identification of molecules](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html) | Benchmark, Transformer, GNN | Advances in Neural Information Processing Systems | [![Star](https://img.shields.io/github/stars/pluskal-lab/MassSpecGym.svg?style=social&label=Star)](https://github.com/pluskal-lab/MassSpecGym) | MassSpecGym |
| [Artificial Intelligence in Spectroscopy: Advancing Chemistry from Prediction to Generation and Beyond](http://arxiv.org/pdf/2502.09897v1) | Neural architectures, ML-empowered solution | arXiv |  | review |
| [BMDMS-NP: A comprehensive ESI-MS/MS spectral library of natural compounds](https://doi.org/10.1016/j.phytochem.2020.112427) | Spectral Library | Phytochemistry | [![Star](https://img.shields.io/github/stars/chalbori/bmdms-np.svg?style=social&label=Star)](https://github.com/chalbori/bmdms-np) | BMDMS-NP, ESI-MS/MS spectral library |
| [Annual Review of Analytical Chemistry Machine Learning in Small-Molecule Mass Spectrometry](https://doi.org/10.1146/annurev-anchem-071224-082157) | Sequence-based Model, Graph-based Model, Deep Learning Model, Siamese Neural Network, Natural Language Processing Model, Transformer, MLP, Random Forest, Bayesian Regularized Neural Network, XGBoost, Light Gradient Boosting Machine, CNN, SVR | Annual Review of Analytical Chemistry |  | Review Paper, SELFIES, Graph Neural Networks, MPNNs, SchNet, DimeNet++, ComENet, DeepMASS, MS2DeepScore, Spec2Vec, CLERMS, NEIMS, MassFormer, 3DMolMS, CFM-ID 4.0, SCARF, ICEBERG, Retip, METLIN-DLM, GNN-RT, DeepGCN-RT, RT-transformer, DNNpwa-TL, MetCCS, AllCCS, CCSBase, CCSP 2.0, DeepCCS, SigmaCCS, AllCCS2, SIRIUS, BUDDY,  mass spectrometry analysis |
| [Critical Assessment of Small Molecule Identification](http://casmi-contest.org/2016/proceedings.shtml.html) | Competation, Benchmark |Phytochemistry Letters ||CASMI 2012, 2013, 2014, 2016, 2017, 2022, [project website](http://casmi-contest.org/2016/challenges-cat2+3.shtml.html)|
| [Insights into predicting small molecule retention times in liquid chromatography using deep learning](https://doi.org/10.1186/s13321-024-00905-1) | SVM, Deep Learning, Transformer, GNN, RF, MLP, DNN, CNN, RNN | Journal of Cheminformatics | [![Star](https://img.shields.io/github/stars/LiuLime/PredRT_review_2024.svg?style=social&label=Star)](https://github.com/LiuLime/PredRT_review_2024.git) | CSI-FingerID, SIRIUS 4, MSNovelist, MassGenie, Smiles-Bert, Smiles transformer, Chemformer, PredRet, GCN, RGCN, MPNN, GIN, DNNpwa-TL, CMM-RT, 1D CNN-TL, AWD-LSTM, TransformerXL, MDC-ANN, retention time_GNN, QGeGNN, RT-transformer, mt-QSRR, MultiConditionRT, HighResNPS, GNN-RT-TL, Retip, SMRT, GNN-TL,  review paper |
| [Quantum chemical electron impact mass spectrum prediction for de novo structure elucidation: Assessment against experimental reference data and comparison to competitive fragmentation modeling](https://doi.org/10.1002/qua.25460) | Quantum Chemical Model, Expert system, Spectrum Calculator | International Journal of Quantum Chemistry |  | QCEIMS, CFM-EI, comparison between first principle simulation and expert system |
| [FragHub: A Mass Spectral Library Data Integration Workflow](https://doi.org/10.1021/acs.analchem.4c02219.s001) | Workflow | Analytical Chemistry  | [![Star](https://img.shields.io/github/stars/eMetaboHUB/FragHub.svg?style=social&label=Star)](https://github.com/eMetaboHUB/FragHub) | FragHub, mass spectral lib integration |
| [Evaluation of the performance of a tandem mass spectral library with mass spectral data extracted from literature](https://doi.org/10.1002/dta.341) | Database | Drug Testing and Analysis |  | MSforID, mass spectrum analysis |
| [Comparative Evaluation of Electron Ionization Mass Spectral Prediction Methods](https://pubs.acs.org/doi/10.1021/jasms.3c00059) | Quantum Chemistry, Machine Learning, Algorithm |  |  | QCEIMS, CFM-EI, NEIMS comparison |
| [Computational mass spectrometry for small-molecule fragmentation](https://doi.org/10.1016/j.trac.2013.09.008) | General models of fragmentation, Simulation Software, Fragmentation Prediction Software, Machine Learning, Heuristic Method, Combination of MetFrag and spectral library search, Classifier, Kernel-based method, Combinatorial Optimization Model | TrAC Trends in Analytical Chemistry |  | DENDRAL, Mass Frontier 4, Mass Frontier 6, ACD Fragmenter, ISIS, MetFrag, MetFusion, Varmuza feature-based classification approach, Heinonen et al. kernel-based approach, Fragmentation Trees, mass spectrometry fragmentation, review paper |
| [Searching molecular structure databases using tandem MS data: are we there yet?](https://doi.org/10.1016/j.cbpa.2016.12.010) | Automated Method, Machine Learning | Current Opinion in Chemical Biology |  | CFM-ID, MetFrag, MAGMa, FingerID, CSI:FingerID, MetFrag2.2, MAGMa+, MSFINDER, MIDAS, IOKR version of CSI:FingerID, metabolite identification, review paper |
| [Unsupervised machine learning for exploratory data analysis in imaging mass spectrometry](https://doi.org/10.1002/mas.21602) | Unsupervised machine learning | Mass Spectrometry Reviews |  | PCA, IMS data analysis,  Project Link: [BioMap](https://www.ms-imaging.org/biomap/), review paper |
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
