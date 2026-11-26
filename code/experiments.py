import json
import sys
from pathlib import Path
from fractions import Fraction
from time import perf_counter
import networkx as nx

from safezones import BARBELL_PARAMS, BROOM_PARAMS, BRIDGEBLOCKS, LAYER_PARAMS
from safezones import centralities, exact, families, real_counts, simulate, strengths

POLICIES = ["weak", "strong", "bc", "bcw", "source-aware", "threshold"]
SEEDSTART = {"bridge blocks":23,"trees":27,"brooms":8,"layered":13,"barbells":19}

def metrics(records):
    rows = [row for row in records if row["exact"]>0]
    result = {"denominator":len(rows),"zero_records":len(records)-len(rows)}
    for policy in POLICIES:
        ratios = [Fraction(row[policy],row["exact"]) for row in rows]
        assert all(0<=ratio<=1 for ratio in ratios)
        result[policy] = {
            "optimal":100*sum(ratio==1 for ratio in ratios)/len(ratios) if ratios else None,
            "mean":float(sum(ratios)/len(ratios)) if ratios else None,
            "minimum":float(min(ratios)) if ratios else None,
        }
    return result

def synthetic():
    started = perf_counter()
    records = []
    pairs = []
    laura = []
    pair_counts = {}
    zero_counts = {}
    topology_index = 0
    topology_manifest = []
    for family, graphs in families():
        pair_counts[family] = 0
        zero_counts[family] = 0
        for position,(label,graph) in enumerate(graphs):
            strengthindex = SEEDSTART[family]+position
            istree = nx.is_tree(graph)
            topology_manifest.append({
                "index": topology_index,
                "strength_index":strengthindex,
                "family": family,
                "label": label,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            })
            exact_values = {source: exact(graph, source) for source in graph.nodes()}
            for source,value in exact_values.items():
                pair = dict(family=family,label=label,source=source,exact=value,tree=istree)
                if istree or family=="layered":
                    pair["thresholdopt"] = exact(graph,source,threshold_only=True)
                pairs.append(pair)
            pair_counts[family] += graph.number_of_nodes()
            zero_counts[family] += sum(value == 0 for value in exact_values.values())
            for realization in range(8):
                values = strengths(graph, strengthindex, realization)
                bc, bcw = centralities(graph, values)
                for source in graph.nodes():
                    row = {
                        "family": family,
                        "label": label,
                        "topology_index": topology_index,
                        "source": source,
                        "realization": realization,
                        "strength_index":strengthindex,
                        "weight_seed": 20260929 + 8 * strengthindex + realization,
                        "exact": exact_values[source],
                        "tree":istree,
                    }
                    for policy in POLICIES:
                        row[policy] = simulate(graph, source, policy, values, bc, bcw)
                    assert all(0<=row[policy]<=row["exact"] for policy in POLICIES)
                    if istree and row["exact"]>0:
                        assert 2*row["threshold"]>=row["exact"],row
                    if "m" in graph.graph and source==graph.graph["source"]:
                        m = graph.graph["m"]
                        assert row["exact"]==2*m and row["threshold"]==m+2
                        assert Fraction(row["threshold"],row["exact"])==Fraction(m+2,2*m)
                        if realization==0:
                            laura.append(dict(label=label,m=m,source=source,exact=row["exact"],
                                              threshold=row["threshold"],ratio=str(Fraction(m+2,2*m))))
                    records.append(row)
            topology_index += 1

    nontrivial = [row for row in records if row["exact"] > 0]
    families_order = list(pair_counts)
    summary = {}
    for family in ["overall", *families_order]:
        rows = records if family == "overall" else [
            row for row in records if row["family"] == family
        ]
        summary[family] = metrics(rows)

    assert all(pair["thresholdopt"]==pair["exact"] for pair in pairs if pair["tree"])
    failures = []
    for pair in pairs:
        if pair["family"]=="layered" and pair["thresholdopt"]<pair["exact"]:
            matches = [row for row in records if row["label"]==pair["label"] and row["source"]==pair["source"]]
            assert len(matches)==8 and all(row["threshold"]==0 for row in matches)
            failures.append(dict(pair,threshold=0))

    layer_graph = next(graph for family, graphs in families() if family == "layered"
                       for label, graph in graphs if label == "layered[1, 2, 2, 1]")
    layer_values = strengths(layer_graph, 14, 0)
    layer_bc, layer_bcw = centralities(layer_graph, layer_values)
    exact_value, exact_cuts = exact(layer_graph, 0, trace=True)
    threshold_value, threshold_cuts, _ = simulate(
        layer_graph, 0, "threshold", layer_values, layer_bc, layer_bcw, trace=True
    )

    result = {
        "topologies": len(topology_manifest),
        "topology_source_pairs": sum(pair_counts.values()),
        "weighted_records": len(records),
        "zero_pairs": sum(zero_counts.values()),
        "nontrivial_weighted_records": len(nontrivial),
        "pair_counts": pair_counts,
        "zero_counts": zero_counts,
        "topology_manifest": topology_manifest,
        "parameters": {
            "bridge_blocks": BRIDGEBLOCKS,
            "brooms": BROOM_PARAMS,
            "layers": LAYER_PARAMS,
            "barbells": BARBELL_PARAMS,
            "strength_distribution": "Uniform(1,10)",
            "weight_seed_formula": "20260929 + 8*strength_index + realization",
            "strength_index_starts":SEEDSTART,
        },
        "summary": summary,
        "records":records,
        "pairs":pairs,
        "laura":laura,
        "layered_failures":failures,
        "layered_failure": {
            "label": "layered[1, 2, 2, 1]",
            "source": 0,
            "exact": exact_value,
            "exact_cuts": exact_cuts,
            "threshold": threshold_value,
            "threshold_cuts": threshold_cuts,
        },
    }
    result["seconds"] = perf_counter()-started
    return result


def real(data_dir):
    base = Path(data_dir)/"raw"
    names = {
        "high_school": "contact-high-school-proj-graph.tar.gz",
        "primary_school": "contact-primary-school-proj-graph.tar.gz",
    }
    from safezones import load_projected
    out = {}
    for key, name in names.items():
        path = base / name
        graph = load_projected(path)
        out[key] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "saved_counts": real_counts(graph),
        }
    return out


def print_synthetic(result):
    print(
        f"topologies={result['topologies']} "
        f"pairs={result['topology_source_pairs']} "
        f"records={result['weighted_records']} "
        f"zero_pairs={result['zero_pairs']} "
        f"nontrivial={result['nontrivial_weighted_records']}"
    )
    print("family          topologies pairs records zero-pairs nontrivial")
    for family,pairs in result["pair_counts"].items():
        count = sum(row["family"]==family for row in result["topology_manifest"])
        print(f"{family:<15} {count:10} {pairs:5} {8*pairs:7} {result['zero_counts'][family]:10} {result['summary'][family]['denominator']:10}")
    for metric,title in (("optimal","EXACT-OPTIMALITY (%)"),("mean","MEAN PROTECTION RATIO"),("minimum","MINIMUM PROTECTION RATIO")):
        print(title)
        print(f"{'family':<15} "+" ".join(f"{policy:>12}" for policy in POLICIES))
        for family,values in result["summary"].items():
            print(f"{family:<15} "+" ".join(f"{values[policy][metric]:12.6f}" for policy in POLICIES))
    for row in result["topology_manifest"]:
        print(f"Topology: {row['family']}, {row['label']}; n={row['nodes']}, m={row['edges']}; strength index={row['strength_index']}")
    for row in result["laura"]:
        print(f"Laura: m={row['m']}; source={row['source']}; TBC={row['threshold']}; OPT={row['exact']}; ratio={row['ratio']}")
    trees = [row for row in result["records"] if row["tree"] and row["exact"]>0]
    print(f"Actual trees: {len(trees)} nontrivial weighted records; all Threshold BC ratios >= 1/2")
    for row in result["layered_failures"]:
        print(f"Layered failure: {row['label']}; source={row['source']}; OPT={row['exact']}; threshold-only OPT={row['thresholdopt']}; TBC={row['threshold']}; all 8 realizations")
    print("parameters=" + json.dumps(result["parameters"], separators=(",", ":")))
    print("layered_failure=" + json.dumps(result["layered_failure"], separators=(",", ":")))
    print(f"Synthetic wall time: {result['seconds']:.6f} seconds")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "synthetic":
        print_synthetic(synthetic())
        return
    if mode == "real":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        print(json.dumps(real(data_dir), indent=2, sort_keys=True))
        return
    if mode == "all":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        result = synthetic()
        print_synthetic(result)
        print(json.dumps(real(data_dir), indent=2, sort_keys=True))
        return
    raise SystemExit("usage: python code/experiments.py synthetic|real|all [data_dir]")


if __name__ == "__main__":
    main()
