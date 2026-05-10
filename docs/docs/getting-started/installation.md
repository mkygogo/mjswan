---
icon: octicons/download-16
---

# Installation

mjswan can be installed as a Python package (the primary workflow) or as an npm package for JavaScript/TypeScript projects.

<div class="grid cards" markdown>

-   [:simple-python: &nbsp; __Python package__](#python-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via pip to quickly build and share interactive MuJoCo simulations

-   [:simple-javascript: &nbsp; __JavaScript package__](#javascript-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via npm for custom web applications with TypeScript support

-   [:simple-github: &nbsp; __GitHub Source__](#github-source){ style="text-decoration: none; color: inherit;" }

    ---

    Clone the repository for development and contributing

>   :simple-docker: &nbsp; __Docker / Cluster__
>   ---
>   Not supported.

</div>

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.12 (3.13+ not yet supported) |
| Platform | macOS (Apple Silicon) or Linux (x86-64) |
| Browser | Any modern browser with WebAssembly and WebGL |
| Node.js | 20+ (npm installation only) |

!!! note "Python 3.13"
    A transitive dependency (`labmaze`, pulled in via MyoSuite) does not yet publish a Python 3.13 wheel. Until it does, mjswan requires Python ≤ 3.12.

## Python Installation

```bash
pip install mjswan
```

Install extra dependency sets as needed:

```bash
pip install 'mjswan[dev]'       # type checking, linting, test tools
pip install 'mjswan[examples]'  # dependencies for the bundled examples
```

The `examples` extra pulls in MyoSuite, MuJoCo Playground, robot_descriptions, and related packages. It can take several minutes to install and requires Python ≤ 3.12.

## JavaScript Installation

```bash
npm install mjswan
```

Or with yarn:

```bash
yarn add mjswan
```

The npm package provides the compiled browser-side runtime. It is independent of the Python package and has no Python dependency.

## GitHub Source

Clone the repository and install all dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ttktjmt/mjswan.git
cd mjswan
uv sync --all-extras
```

To run the bundled demo after cloning:

```bash
mjswan demo          # runs the default demo
mjswan demo --list   # see all available demos
```
