# genomicsdx
# 🧬 GenomicsDx DHF Command Center

## 1. Introduction

The GenomicsDx DHF Command Center is a comprehensive Streamlit application designed to serve as the primary Quality Management System (QMS) and development dashboard for a **Class III, PMA-required, breakthrough-designated Multi-Cancer Early Detection (MCED) genomic diagnostic service**.

This tool is built by a Subject Matter Expert (SME) to manage a Design History File (DHF) in accordance with key medical device regulations and standards, including:
- **21 CFR 820.30** (Design Controls)
- **ISO 13485:2016** (Quality Management Systems)
- **ISO 14971:2019** (Risk Management)
- **ISO 62304:2006** (Software Lifecycle Processes)
- **CLIA / ISO 15189** (Clinical Laboratory Standards)

It provides real-time, interactive insights into all facets of the product development lifecycle, from initial planning and requirements management to analytical validation, clinical trials, and operational readiness.

## 2. Key Features

- **Executive Health Dashboard:** High-level KPIs tracking program health against schedule, quality, and compliance goals.
- **DHF Explorer:** A tabbed interface to manage all sections of the Design History File as mandated by 21 CFR 820.30.
- **Advanced Analytics:** Includes a live, multi-level Traceability Matrix and a centralized Compliance Action & CAPA Tracker.
- **Statistical Workbench:** A suite of interactive tools for performing rigorous statistical analysis essential for Analytical Validation (e.g., Levey-Jennings, DOE, Gauge R&R, Equivalence Testing).
- **ML & Bioinformatics Lab:** Tools for validating and interpreting machine learning models, including classifier explainability with SHAP and predictive models for lab operations.
- **Regulatory Guide:** Embedded educational content explaining the complex regulatory landscape for a genomic IVD service.

## 3. Setup and Installation

To run this application, you must have Python 3.9+ installed. It is highly recommended to use a virtual environment.

**Step 1: Clone the repository (if applicable)**
```bash
git clone <repository-url>
cd genomicsdx-command-center
