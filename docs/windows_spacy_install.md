# Troubleshooting spaCy installation on Windows

The error log shown below indicates that `spacy==3.8.11` tried to compile its
`blis` dependency from source and failed with `WinError 2` when invoking
`clang.exe`:

```
Building wheel for blis (pyproject.toml): finished with status 'error'
error: [WinError 2] The system cannot find the file specified
```

This happens when pip cannot find a pre-built wheel that matches your Python
version/architecture and therefore falls back to compiling C extensions without
the required build toolchain. To avoid the compilation step and install the
pre-built wheels shipped by spaCy, try the following commands in a clean shell
or virtual environment on Windows:

```powershell
# 1) Upgrade pip and wheel tooling so pip can locate binary wheels
python -m pip install --upgrade pip setuptools wheel

# 2) Prefer binary wheels for spaCy and its compiled dependencies
python -m pip install "spacy>=3.8,<3.9" --prefer-binary
```

If you need to pin exactly 3.8.11 you can combine the preference for wheels with
an explicit version:

```powershell
python -m pip install "spacy==3.8.11" --prefer-binary
```

If pip still attempts to build from source (for example, on unusual Python
versions without matching wheels), install the Microsoft "Build Tools for
Visual Studio" with the C++ workload so that `blis`, `thinc`, and other C
extensions can compile:

1. Download the installer from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>.
2. Select the **Desktop development with C++** workload.
3. Re-run the installation commands above after the toolchain is available.

These steps allow `blis` to compile successfully when wheels are unavailable or
help pip pick the official spaCy wheels when they exist, preventing the
`subprocess-exited-with-error` failure.
