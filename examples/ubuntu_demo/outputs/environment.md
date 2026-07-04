# Ubuntu GPU Proof Environment

## System
Linux rigel.cc.gatech.edu 6.17.0-23-generic #23~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 14 16:11:48 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.3 LTS
Release:	24.04
Codename:	noble

## GPU
NVIDIA GeForce RTX 5090, 580.105.08, 32607 MiB

## CUDA
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

## Python and Conda
Python 3.13.9
conda 25.11.1

## PyTorch CUDA: sam3
torch: 2.10.0.dev20251208+cu128
cuda: True
device: NVIDIA GeForce RTX 5090

## PyTorch CUDA: sam3d-objects
torch: 2.8.0+cu128
cuda: True
device: NVIDIA GeForce RTX 5090

## Model checkout commits
SAM3_ROOT=<video2sim2real>/Scene_reconstruction/sam3
757bbb0206a0b68bee81b17d7eb4877177025b2f
SAM3D_ROOT=<video2sim2real>/Scene_reconstruction/sam-3d-objects
afdf6a31522d038c44c68a0bb57aa68827380797
