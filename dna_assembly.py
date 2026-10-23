"""
Lab 4 - DNA Sequence Assembly
1. Uses SARS-CoV-2 genome (NC_045512.2) downloaded from NCBI (FASTA)
2. Takes 20000 random samples of 100-150 bases each
3. Stores samples in a list
4. Rebuilds (assembles) the genome using overlap-based assembly
5. Identifies the main problem with this approach

Run:  python dna_assembly.py genome.fasta
"""

import sys
import random

N_SAMPLES = 20000
MIN_LEN   = 100
MAX_LEN   = 150
DEFAULT_FASTA = "genome.fasta"


# -- 1. Read FASTA ------------------------------------------------------------
def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">") or not line:
                continue
            seq.append(line.upper())
    return "".join(seq)


# -- 2. Take random samples ---------------------------------------------------
def take_samples(seq, n=N_SAMPLES, min_len=MIN_LEN, max_len=MAX_LEN):
    samples = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        start  = random.randint(0, len(seq) - length)
        samples.append(seq[start : start + length])
    return samples


# -- 3. Simple greedy overlap assembly ----------------------------------------
def overlap(a, b, min_overlap=20):
    """Return length of longest suffix of a that matches a prefix of b."""
    start = max(0, len(a) - len(b))
    best  = 0
    for i in range(start, len(a) - min_overlap + 1):
        if b.startswith(a[i:]):
            best = len(a) - i
            break                      # first match from the left is the longest
    return best


def greedy_assembly(reads, min_overlap=20, max_steps=500):
    """Greedy: repeatedly merge the pair with the biggest overlap."""
    contigs = list(reads)
    merged  = True
    steps   = 0
    while merged and steps < max_steps:
        merged = False
        best_score, best_i, best_j = 0, -1, -1
        # only check a random subset of contigs to keep it fast
        subset = random.sample(range(len(contigs)), min(50, len(contigs)))
        for i in subset:
            for j in subset:
                if i == j:
                    continue
                ov = overlap(contigs[i], contigs[j], min_overlap)
                if ov > best_score:
                    best_score, best_i, best_j = ov, i, j
        if best_score >= min_overlap:
            merged_seq = contigs[best_i] + contigs[best_j][best_score:]
            for idx in sorted([best_i, best_j], reverse=True):
                contigs.pop(idx)
            contigs.append(merged_seq)
            merged = True
        steps += 1
    return contigs


# -- 4. Chart -----------------------------------------------------------------
def show_chart(samples, contigs, genome_len):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("DNA Assembly Results - SARS-CoV-2", fontsize=14, fontweight="bold")

    lengths = [len(s) for s in samples]
    axes[0].hist(lengths, bins=25, color="steelblue", edgecolor="white")
    axes[0].set_title(f"Sample Length Distribution ({len(samples)} samples)")
    axes[0].set_xlabel("Length (bases)")
    axes[0].set_ylabel("Count")

    contig_lengths = sorted([len(c) for c in contigs], reverse=True)
    axes[1].bar(range(1, len(contig_lengths) + 1), contig_lengths, color="tomato")
    axes[1].set_title(f"Contig Lengths ({len(contigs)} contigs)")
    axes[1].set_xlabel("Contig rank")
    axes[1].set_ylabel("Length (bases)")

    recovered = min(max(contig_lengths), genome_len)
    axes[2].bar(["Longest contig", "Not recovered"],
                [recovered, genome_len - recovered],
                color=["steelblue", "tomato"])
    axes[2].bar_label(axes[2].containers[0])
    axes[2].set_title("Genome Recovery")
    axes[2].set_ylabel("Bases")

    plt.tight_layout()
    plt.savefig("assembly_results.png", dpi=150)
    print("\nChart saved as assembly_results.png (close the chart window to finish)")
    plt.show()


# -- Main ---------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FASTA

    print("Reading genome...")
    genome = read_fasta(path)
    print(f"Genome length       : {len(genome)} bases")

    random.seed(42)
    print(f"\nTaking {N_SAMPLES} random samples ({MIN_LEN}-{MAX_LEN} bases each)...")
    samples = take_samples(genome)
    print(f"Samples collected   : {len(samples)}")
    print(f"Sample length range : {min(len(s) for s in samples)} - {max(len(s) for s in samples)} bases")
    total_bases = sum(len(s) for s in samples)
    avg_len = total_bases / len(samples)
    avg_cov = total_bases / len(genome)
    print(f"Average sample len  : {avg_len:.1f} bases")
    print(f"Total bases sampled : {total_bases}")
    print(f"Average coverage    : {avg_cov:.1f}x")

    print("\nFirst 3 samples:")
    for i, s in enumerate(samples[:3]):
        print(f"  Sample {i+1} (len={len(s)}): {s[:40]}...")

    n_asm = 100   # greedy assembly on all 20000 reads would take far too long
    print(f"\nAttempting greedy overlap assembly on {n_asm} reads (max 500 merge steps)...")
    contigs = greedy_assembly(samples[:n_asm], min_overlap=20)
    contigs.sort(key=len, reverse=True)
    print(f"Contigs assembled   : {len(contigs)}")
    print(f"Longest contig      : {len(contigs[0])} bases")
    print(f"Original genome     : {len(genome)} bases")
    print(f"Recovery            : {len(contigs[0]) / len(genome) * 100:.1f}% of genome in longest contig")

    n = len(samples)
    print("\n" + "=" * 60)
    print("MAIN PROBLEMS WITH THIS ALGORITHM:")
    print("=" * 60)
    problems = [
        ("1. Repeat regions",
         "   Reads that fall inside repeated sequences cannot be placed\n"
         "   correctly. The assembler may merge them in the wrong order,\n"
         "   creating errors or collapsing repeats."),
        ("2. Coverage gaps",
         f"   With {n} reads of ~{avg_len:.0f} bases over a {len(genome)}-base genome,\n"
         f"   average coverage is ~{avg_cov:.0f}x. Coverage is random, so some\n"
         "   regions can still get few or 0 reads, breaking the assembly."),
        ("3. Greedy assembly is not optimal",
         "   A greedy approach picks the locally best merge at each step.\n"
         "   This may not lead to the globally correct assembly.\n"
         "   Finding the optimal solution is NP-hard."),
        ("4. Short reads = short overlaps",
         f"   {MIN_LEN}-{MAX_LEN} base reads produce short overlaps (~20 bases).\n"
         "   Short overlaps are more likely to be wrong matches,\n"
         "   leading to mis-assemblies."),
        ("5. Scalability",
         f"   Checking all pairs of reads is O(n^2). With {n} reads\n"
         f"   that is {n * n:,} comparisons, very slow for large genomes."),
    ]
    for title, desc in problems:
        print(f"\n{title}")
        print(desc)

    show_chart(samples, contigs, len(genome))


if __name__ == "__main__":
    main()