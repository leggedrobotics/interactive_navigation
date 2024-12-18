# Template for Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository serves as a template for building projects or extensions based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab


### Installation


- Throughout the repository, the name `interactive_navigation` only serves as an example and we provide a script to rename all the references to it automatically:

```
# Rename all occurrences of interactive_navigation (in files/directories) to your_fancy_extension_name
python scripts/rename_template.py your_fancy_extension_name
```

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Using a python interpreter that has Isaac Lab installed, install the library

```
cd exts/interactive_navigation
python -m pip install -e .
```

#### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Lab installation.

If everything executes correctly, it should create a file .python.env in the .vscode directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


#### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `exts/interactive_navigation/interactive_navigation/ui_extension_example.py`. For more information on UI extensions, enable and check out the source code of the `omni.isaac.ui_template` extension and refer to the introduction on [Isaac Sim Workflows 1.2.3. GUI](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html#gui).

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `IsaacLabExtensionTemplate/exts`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source/extensions`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Docker Setup for Extension and rsl_rl

All changes are done in `IsaacLab-Internal/docker`
rsl_rl and the extension should be installed besides IsaacLab:

```
├── Project/
│ ├── IsaacLab-Internal/
│ ├── rsl_rl/
│ ├── interactive_navigation/
```



#### Step 1
In `IsaacLab-Internal/docker/.env.base` add the paths to rsl_rl and the extension 

```docker
# Local rsl_rl path
LOCAL_RSL_RL_RELATIVE_PATH=../../rsl_rl/rsl_rl

# Locale extension path
LOCAL_EXTENSION_RELATIVE_PATH=../../interactive_navigation
```

#### Step 2
In `IsaacLab-Internal/docker/docker-compose.yaml`, bind the two modules
```yaml
  - type: bind
    source: ${LOCAL_RSL_RL_RELATIVE_PATH}
    target: ${DOCKER_ISAACLAB_PATH}/rsl_rl
  - type: bind
    source: ${LOCAL_EXTENSION_RELATIVE_PATH}
    target: ${DOCKER_ISAACLAB_PATH}/isaaclab_extension
```

Change the context and the docker file path from
```yaml
      context: ../
      dockerfile: docker/Dockerfile.base
```
to
```yaml
      context: ../..
      dockerfile: IsaacLab-Internal/docker/Dockerfile.base
```

#### Step 3
In `IsaacLab-Internal/docker/Dockerfile.base` change paths to image from
```docker
COPY ../ ${ISAACLAB_PATH}
```
to
```docker
COPY IsaacLab-Internal/ ${ISAACLAB_PATH}
# Copy rsl_rl into the image
COPY rsl_rl/ ${ISAACLAB_PATH}/rsl_rl
# Copy extension into the image
COPY interactive_navigation/ ${ISAACLAB_PATH}/isaaclab_extension
```

and install the additional modules after installing Isaac Lab dependencies
```docker
# Upgrade pip, setuptools and wheel to the latest version
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip setuptools wheel

# Install local rsl_rl module
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e ${ISAACLAB_PATH}/rsl_rl

# Install local isaaclab_extension module
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e ${ISAACLAB_PATH}/isaaclab_extension/exts/interactive_navigation

```

## Cluster
Student pc documentation: https://student-pc-docs.leggedrobotics.com/docs/apptainer.html

Isaac Lab internal documentation: https://leggedrobotics.github.io/IsaacLab-Internal/source/deployment/cluster.html

#### Step 1
Within `/scratch/user`  install the latest apptainer  version (1.3.5)
```bash
cd /scratch/${USER}
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | \
    bash -s - install-dir
```
And add the path to the end of the .bashrc :
```bash
echo "export PATH=/scratch/${USER}/install-dir/bin:\$PATH" >> ~/.bashrc
```
#### Step 2
Configure cluster parameters in `IsaacLab-Internal/docker/cluster/.env.cluster` 
substitute `username` with your username

```bash
###
# Cluster specific settings
###

# Job scheduler used by cluster.
# Currently supports PBS and SLURM
CLUSTER_JOB_SCHEDULER=SLURM
# Docker cache dir for Isaac Sim (has to end on docker-isaac-sim)
# e.g. /cluster/scratch/$USER/docker-isaac-sim
CLUSTER_ISAAC_SIM_CACHE_DIR=/cluster/scratch/username/docker-isaac-sim
# Isaac Lab directory on the cluster (has to end on isaaclab)
# e.g. /cluster/home/$USER/isaaclab
CLUSTER_ISAACLAB_DIR=/cluster/home/username/isaaclab
# Cluster login
CLUSTER_LOGIN=username@euler.ethz.ch
# Cluster scratch directory to store the SIF file
# e.g. /cluster/scratch/$USER
CLUSTER_SIF_PATH=/cluster/scratch/username
# Remove the temporary isaaclab code copy after the job is done
REMOVE_CODE_COPY_AFTER_JOB=false
# Python executable within Isaac Lab directory to run with the submitted job
# CLUSTER_PYTHON_EXECUTABLE=source/standalone/workflows/rsl_rl/train.py
# We use the train script of the extension, not the one in IsaacLab
CLUSTER_PYTHON_EXECUTABLE=isaaclab_extension/scripts/rsl_rl/train.py 

```

#### Step 4
Enable ssh key-based authentication for euler.
Generate rsa ssh key:
```bash
ssh-keygen -t rsa -b 4096 -C "email@example.com"
```
Copy key to cluster (with your username)
```bash
ssh-copy-id username@euler.ethz.ch
```
Enter your eth password

Test connection:
```bash
ssh username@euler.ethz.ch
```
This time, no password should be required.


#### Step 5
Export singularity image. Within `IsaacLab-Internal`, run
```bash
./docker/cluster/cluster_interface.sh push
```
this takes up to 20 min.

#### Step 6
Add your email to `docker/cluster/submit_job_slurm.sh` and load eth_proxy, i.e.,
```bash
# e.g., `module load eth_proxy`
module load eth_proxy
```

#### Step 7 Submitting a job
To submit a job, cd into `IsaacLab-Internal` and run
```bash
./docker/cluster/cluster_interface.sh job "argument1" "argument2" ...

```
You can submit multiple jobs in parallel. If you need to update your docker environment, you need to repeat step 5.

