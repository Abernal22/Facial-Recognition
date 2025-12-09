import subprocess
import sys

PACKAGES = [
    "numpy",
    "pandas",
    "opencv-python",
    "tensorflow",
    "scikit-learn",
    "matplotlib",
    "kagglehub",
    "transformers",
    "torch",
    "torchvision",
    "tqdm",
]

def install(package: str):
    """Run `pip install package` with the current Python interpreter."""
    print(f"\n=== Installing {package} ===")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    for pkg in PACKAGES:
        try:
            __import__(pkg.split("==")[0].split(">=")[0])
            print(f"{pkg} already installed, skipping.")
        except ImportError:
            install(pkg)

if __name__ == "__main__":
    main()
