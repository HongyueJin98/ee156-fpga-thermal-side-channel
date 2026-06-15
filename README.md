# ee156-fpga-thermal-side-channel
# FPGA Thermal Side-Channel Analysis

FPGA-based side-channel analysis project completed for EE156 Advanced Topics in Computer Architecture at Tufts University.

## Overview

This project investigates whether encryption algorithms can be identified using FPGA implementation characteristics extracted from Vivado HLS synthesis reports.

The following algorithms were analyzed:

* AES
* PRESENT
* Serpent
* RSA

## Method

RTL-level hardware features were extracted from synthesis reports, including:

* Latency
* Flip-Flop (FF) utilization
* LUT utilization
* Memory utilization

A fully connected neural network (FCNN) implemented in PyTorch was trained to classify encryption algorithms using these hardware features.

## Results

The classifier achieved approximately 90% accuracy in distinguishing different encryption algorithms based solely on FPGA implementation characteristics.

## Repository Structure

```text
scripts/
    Neural network training code
src/
    Encryption algorithms
docs/
    Final project report

```

## Technologies

* FPGA
* Vivado HLS
* Side-Channel Analysis
* Python
* PyTorch
* Machine Learning

## Course

EE156 Advanced Topics in Computer Architecture
Tufts University
