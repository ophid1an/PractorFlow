"""
Check CPU instruction set support for llama-cpp-python compatibility.
"""

import subprocess
import platform

def check_cpu_features():
    """Check which CPU features are available."""
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print()
    
    if platform.system() == "Linux":
        try:
            # Check CPU flags in WSL2
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Extract flags
            for line in cpuinfo.split('\n'):
                if line.startswith('flags'):
                    flags = line.split(':')[1].strip().split()
                    
                    print("CPU Instruction Sets:")
                    print(f"  SSE4.1: {'sse4_1' in flags or 'sse4a' in flags}")
                    print(f"  SSE4.2: {'sse4_2' in flags}")
                    print(f"  AVX:    {'avx' in flags}")
                    print(f"  AVX2:   {'avx2' in flags}")
                    print(f"  AVX512: {'avx512f' in flags}")
                    print()
                    
                    if 'avx2' not in flags:
                        print("⚠️  WARNING: AVX2 not supported!")
                        print("   The precompiled wheel requires AVX2.")
                        print("   You need to build from source with compatible flags.")
                    else:
                        print("✓ CPU supports AVX2 - wheel should work")
                    
                    break
        except Exception as e:
            print(f"Error reading CPU info: {e}")
    else:
        print("CPU check only works on Linux/WSL2")

if __name__ == "__main__":
    check_cpu_features()