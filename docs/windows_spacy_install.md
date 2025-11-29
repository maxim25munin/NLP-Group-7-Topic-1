# spaCy installation workaround on Windows

This guide fixes the `WinError 2`/`error: [WinError 2] The system cannot find the file specified` failure that occurs while `pip` tries to compile the `blis` dependency from source during `spacy<3.9` installation.

## Quick fix: force prebuilt wheels

Binary wheels are published for Windows and avoid the C toolchain requirement entirely. For a minimal CPU-only install of spaCy 3.8.x on Python 3.9, run the following in **PowerShell** or **Command Prompt**:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: "spacy<3.9" "thinc<8.4" "blis<1.4" \
    "murmurhash<1.1" "cymem<2.1" "preshed<3.1" "srsly<3.0" "catalogue<2.1"
```

Key points:

* `--only-binary=:all:` tells `pip` to download wheels and refuse building from source.
* The explicit pins match spaCy 3.8.x requirements so the resolver can fetch compatible wheels.
* If you also need GPU builds, add spaCy's official wheel index: `--extra-index-url https://download.spacy.io/wheels/cu121` (replace `cu121` with your CUDA version).

## Alternative: install the required build tools

If you prefer source builds (or use an environment without wheels), install the Microsoft C++ Build Tools and set the MSVC toolchain as the default:

1. Install **[Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** and select the "Desktop development with C++" workload.
2. Open a **x64 Native Tools Command Prompt** for VS and retry the installation:
   ```cmd
   python -m pip install "spacy<3.9"
   ```

## Verify the install

After installation, confirm that spaCy imports correctly and that the matching pipeline models can be downloaded:

```powershell
python -c "import spacy; print(spacy.__version__)"
python -m spacy download en_core_web_sm
```

If the commands run without compilation errors, the `blis` wheel was installed successfully.
