# FSAE Autonomous Car Sim

A Python simulation environment for prototyping autonomous-vehicle control algorithms for an FSAE (Formula SAE) car, built on a fork of the [F1TENTH Gym](https://github.com/Anna-Kopec/f1tenth_gym).

## Why F1TENTH?

F1TENTH is a lightweight simulator that handles the vehicle physics, so you can focus on implementing the control algorithms themselves.

- **Python only.** No heavy rendering engines (Unreal, Unity) are required, unlike many other simulators.
- **Easy to set up.** A virtualenv and a single `pip install` are enough to get running.
- **Well suited to prototyping.** Fast iteration on planning and control before moving to higher-fidelity tools.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3 | Python 3.8 is required on Windows (upstream library-support limitation). |
| `git` | To clone the simulator. |
| `virtualenv` | Install with `pip install virtualenv` (macOS: `pip3 install virtualenv`). |

## Quick Start

Commands below run in macOS Terminal or Windows PowerShell. Where a step has one command, it is identical on both systems.

### 1. Clone the repository

```bash
git clone https://github.com/Anna-Kopec/f1tenth_gym
cd f1tenth_gym
```

### 2. Create the virtual environment

From inside `f1tenth_gym`:

```bash
virtualenv gym_env
```

### 3. Activate the virtual environment and install the simulator

| macOS | Windows (PowerShell) |
|---|---|
| `source gym_env/bin/activate` | `gym_env\Scripts\Activate.ps1` |

Then, on both systems:

```bash
pip install -e .
```

Use `deactivate` to exit the environment. On Windows Command Prompt (cmd), activate with `gym_env\Scripts\activate.bat`.

### 4. Run the genetic simulation

| macOS | Windows (PowerShell) |
|---|---|
| `python3 genetic_sim/testing_env.py` | `python genetic_sim\testing_env.py` |

Run from the repository root. If the script cannot find its map or data files, `cd genetic_sim` first and run `testing_env.py` from there.

## Documentation

### Read Write-ups
Formal written documentation for this project can be found in docs/writeups. 
* [Autonomous Car Guidelines](<docs/writeups/Autonomous Car Guidelines.pdf>): UConn FSAE's autonomous vehicle initiative overview
* [CSE 4820 Project Proposal](docs/writeups/CSE_4820_Course_Project_Proposal.pdf): Our team's project proposal (this is also doubling as a class project) 

### Viewing Doxygen Code Documentation

Open `docs/doxygen/html/index.html` in a browser.

| macOS | Windows (PowerShell) |
|---|---|
| `open docs/doxygen/html/index.html` | `start docs\doxygen\html\index.html` |

You can also double-click `index.html` in Finder or File Explorer. If search or navigation breaks when opened as a local file, serve the folder and visit `http://localhost:8000`:

| macOS | Windows (PowerShell) |
|---|---|
| `python3 -m http.server 8000 --directory docs/doxygen/html` | `python -m http.server 8000 --directory docs\doxygen\html` |

## Troubleshooting

| Issue | Fix |
|---|---|
| `command not found` / "not recognized" (e.g., `virtualenv`, `git`) | Install the missing prerequisite and make sure it is on your `PATH`. |
| PowerShell: "running scripts is disabled" when activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. |
| Library errors on Windows | Use Python 3.8 (e.g., `py -3.8 -m virtualenv gym_env`). |
| macOS (Big Sur+) with rendering on: `ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework` | Run `pip3 install pyglet==1.5.20`. |
| `f110-gym requires pyglet<1.5, but you have pyglet 1.5.20` | Safe to ignore; the environment should still work. |
