import pkg_resources
import sys

def check_libraries():
    # List of libraries from the project's context
    required_libraries = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scikit-learn', 'tensorflow', 'papaparse', 
        'scipy', 'joblib'
    ]
    
    print("Checking Python environment libraries:")
    print(f"Python version: {sys.version}")
    
    missing_libs = []
    for lib in required_libraries:
        try:
            pkg_resources.get_distribution(lib)
            print(f"✓ {lib} is installed")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {lib} is NOT installed")
            missing_libs.append(lib)
    
    if missing_libs:
        print("\nMissing libraries:")
        print(", ".join(missing_libs))
        print("\nTo install, run: pip install " + " ".join(missing_libs))
    else:
        print("\nAll required libraries are installed!")

if __name__ == "__main__":
    check_libraries()