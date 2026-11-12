"""Ceiling on decode speed: memory bandwidth divided by bytes read per token.

Every generated token reads all the weights once, so
  tokens/s <= bandwidth (bytes/s) / model size (bytes).
"""
import sys

def ceiling(bandwidth_gbs, model_gb):
    return bandwidth_gbs / model_gb

if __name__ == "__main__":
    bw = float(sys.argv[1])          # GB/s, from the vendor's page
    for name, gb in [("0.6B Q4_K_M", 0.373), ("4B Q4_K_M", 2.4),
                     ("4B Q8_0", 4.0)]:
        print(f"{name:12s} {gb:5.2f} GB -> at most {ceiling(bw, gb):6.0f} tok/s")
