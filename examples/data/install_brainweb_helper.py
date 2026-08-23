#!/usr/bin/env python3
import sys
import subprocess

def install_brainweb():
    try:
        import brainweb
    except ImportError:
        print("brainweb not found. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "brainweb"])
        try:
            import brainweb
        except ImportError:
            print("Installation of brainweb failed.")
            sys.exit(1)
    print("brainweb is installed.")
    
def install_numba():
    try:
        import numba
    except ImportError:
        print("numba not found. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numba"])
        try:
            import numba
        except ImportError:
            print("Installation of numba failed.")
            print("This doesn't really matter but things will be slower.")
    print("numba is installed.")

def check_sirf_STIR():
    try:
        from sirf import STIR  # checks for sirf.STIR availability
        print("sirf.STIR is installed.")
    except ImportError:
        print("sirf.STIR is not installed. Please install the SIRF package manually.")
        sys.exit(1)

def check_cil():
    try:
        import cil
        print("cil is installed.")
    except ImportError:
        print("cil is not installed. Please install the CIL package manually.")
        sys.exit(1)

def main():
    install_brainweb()
    install_numba()
    check_sirf_STIR()
    check_cil()
    print("All required packages are installed and working.")

if __name__ == "__main__":
    main()
