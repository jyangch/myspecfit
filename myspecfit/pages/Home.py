import streamlit as st


st.set_page_config(
    page_title="Home",
    page_icon="🏠")

st.write("# Welcome to :rainbow[*MySpecFit*] 👋")

st.markdown(
    """
    BaySpec is a Bayesian inference-based spectral fitting tool for multi-dimensional (time and energy) 
    and multi-wavelength (X-ray and gamma-ray) astrophysical data.
    ### Framework:
    - Spectrum class: Setting up each set of spectra and detector response as well as the statistics.
    - Model class: Setting up models for fitting spectra.
    - Fit class: Sampling for model parameters based on Bayesian theory.
    - Analyse class: Analyzing the posterior samples to determine the best-fit parameters and goodness.
    - Plot class: Plotting the observed and model-predicted spectra.
    - Calculate class: Calculating flux based on posterior samples.
    ### Features:
    - Bayesian inference-based: implemented by MCMC (e.g., emcee) or nested sampling (e.g., multinest)
    - Multi-dimensional: enabling the fit of time-evolving spectra with time-involved physical models
    - Multi-wavelength: supporting for the joint fitting to multi-wavelength astronomical spectra
    - Others: simultaneous fitting of multi-spectra and multi-models, freely combining available models and add new model
    ### Available models:
    - Empirical models: pl, cpl, band, sbpl, cband, csbpl, dband, dsbpl...
    - Physical models: bb, hle, mgf, photo, sync, ssc...
    - Absorption models: phabs, tbabs...
    """
    )
