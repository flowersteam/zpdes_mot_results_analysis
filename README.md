# Results Analysis

This repository houses all the scripts necessary for analyzing the data associated with the paper
"AI-ENHANCED MULTI-OBJECT TRACKING: TOWARD INDIVIDUALIZED COGNITIVE TRAINING."
The data will become available alongside the published paper.

## Installation Instructions

To begin, clone this repository using:

First install this repository:
`git clone ...`

This analysis requires Python 3. Make sure you have the correct version installed.  
It is strongly advised to create and utilize a virtual environment to avoid conflicts.

For example, you can set up an environment with Conda:

`conda install --env my_env`

Next, activate your environment and install the necessary dependencies:

`conda activate my_env`
`pip3 install -r requirements.txt`

Your setup is now complete!

## Usage

The project is organized into three primary directories:

- analysis_scripts: This directory contains all scripts for performing the analysis.
  - cognitive_battery: Scripts for analyzing cognitive battery data.
  - questionnaires: Scripts for analyzing questionnaire data.
- data: This directory is for storing all data gathered during our experiments.
- outputs: This directory is for saving all analysis outputs.

- To execute the full analysis, ensure your virtual environment is activated:

`conda activate my_env`

Then, run the main function in Python:

`python main.py`

You should see output in your console indicating that the scripts are executing.

## Config files:

Several files are available to configure different aspects of the analysis:

- Conditions:
- Visual features:
- Fitting params:

