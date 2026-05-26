import json

DATA_DIR = "/workspace/results/app/data"
OUT_DIR  = "/workspace/results/app/output"

# --- Load all input files ---
with open(f"{DATA_DIR}/route.json")           as f: route          = json.load(f)
with open(f"{DATA_DIR}/preferences.json")     as f: preferences    = json.load(f)
with open(f"{DATA_DIR}/local_pref.json")      as f: local_pref     = json.load(f)
with open(f"{DATA_DIR}/relationships.json")   as f: relationships  = json.load(f)
with open(f"{DATA_DIR}/topology.json")        as f: topology       = json.load(f)
with open(f"{DATA_DIR}/route_events.json")    as f: route_events   = json.load(f)
with open(f"{DATA_DIR}/possible_solutions.json") as f: possible_solutions = json.load(f)

# =========================================================
# 1. OSCILLATION DETECTION
#    A routing-preference cycle exists when hub A prefers
#    routes via hub B AND hub B prefers routes via hub A.
# =========================================================
pref_map = {int(asn): int(p["prefer_via"]) for asn, p in preferences.items()}

oscillation_cycle = []
visited = set()
for asn, via in pref_map.items():
    if asn not in visited and via in pref_map and pref_map[via] == asn:
        oscillation_cycle = [asn, via]
        visited.add(asn)
        visited.add(via)
        break   # report the first (and usually only) cycle

oscillation_detected = bool(oscillation_cycle)
affected_ases = sorted(oscillation_cycle) if oscillation_cycle else []

print(f"Oscillation detected: {oscillation_detected}")
print(f"Cycle: {oscillation_cycle}")

# =========================================================
# 2. ROUTE LEAK DETECTION
#    Valley-free violation: routes learned from a provider
#    MUST NOT be re-advertised to a peer (or another provider).
#    We have explicit events in route_events.json.
# =========================================================
# Build a lookup: (from_asn, to_asn) -> relationship type (from perspective of from_asn)
rel_lookup = {}
for r in relationships:
    rel_lookup[(r["from"], r["to"])] = r["type"]

route_leaks = []
for ev in route_events:
    leaker  = ev["advertiser_asn"]
    src     = ev["source_asn"]
    dst     = ev["destination_asn"]
    src_rel = rel_lookup.get((leaker, src))
    dst_rel = rel_lookup.get((leaker, dst))
    if src_rel is None:
        src_rel = ev.get("source_type")
    if dst_rel is None:
        dst_rel = ev.get("destination_type")
    # Valley-free: a route learned from a provider must only go to customers
    if src_rel == "provider" and dst_rel in ("peer", "provider"):
        route_leaks.append({
            "leaker_as":        leaker,
            "source_as":        src,
            "destination_as":   dst,
            "source_type":      src_rel,
            "destination_type": dst_rel
        })

route_leak_detected = bool(route_leaks)
print(f"Route leak detected: {route_leak_detected}, leaks: {route_leaks}")

# =========================================================
# 3. EVALUATE SOLUTIONS
#    oscillation_resolved: solution must break the preference
#      cycle (stop at least one hub preferring the other).
#    route_leak_resolved : solution must stop hub1/hub2 from
#      advertising provider routes to hub2/hub1 (egress action).
# =========================================================
#
# We map each solution string to (osc_resolved, leak_resolved)
# using BGP/Azure Virtual WAN domain knowledge.
#
solution_verdicts = {
    # --- Ingress filter on hub1 for prefixes from hub2
    # Does NOT stop hub1 from advertising provider routes to hub2
    # Does NOT break the mutual preference cycle
    "Apply route filter on vhubvnet1 to only accept prefixes 10.2.0.0/16 from vhubvnet2":
        (False, False),

    # --- no-export community on provider routes at hub1
    # Prevents hub1 from advertising provider routes to eBGP peer hub2 → stops leak
    # Hub2 can no longer prefer hub1 for those routes → cycle broken
    "Configure route policy on vhubvnet1 to enforce no-export of provider routes to peer vhubvnet2 by bgp community":
        (True, True),

    # --- shorter AS-PATH / MED preference on hub1
    # Direct VirtualWAN path is shorter AS-PATH than hub2 path
    # Hub1 would prefer VirtualWAN → hub1 side of cycle broken → oscillation resolved
    # Does NOT stop hub1 from advertising to hub2
    "Configure vhubvnet1 to prefer shorter AS-PATH length or use MED value 100 for route selection":
        (True, False),

    # --- RPKI origin validation on hub2
    # Validates origin AS for the advertised prefix; does not affect route preferences
    # or advertising policy between hubs
    "Enable RPKI origin validation on vhubvnet2 to verify prefix 10.2.1.0/24 originates from ASN 65007 (VNET4)":
        (False, False),

    # --- Ingress AS-PATH filter on hub2 rejecting routes containing 65001
    # Hub2 rejects leaked routes → hub2 cannot prefer hub1 for provider routes → cycle broken
    # Hub1 still advertises (egress policy unchanged) → leak not stopped per definition
    "Enable ingress filtering on vhubvnet2 (ASN 65003) to reject routes with AS_PATH containing Virtual WAN ASN 65001 when received from peer vhubvnet1":
        (True, False),

    # --- UDR override on hub1 to prefer a specific next-hop
    # Overrides hub1's preference away from hub2 → breaks hub1 side of preference cycle
    # Does NOT stop hub1 from advertising provider routes to hub2
    "Configure user defined route override on vhubvnet1 defaultRouteTable to prefer specific next-hop":
        (True, False),

    # --- Virtual WAN Hub Routing Intent (all hub-to-hub via WAN only)
    # Forces all inter-hub traffic through VirtualWAN, effectively removing
    # the direct peer advertisement path → breaks cycle and stops leak
    "Enable Virtual WAN hub routing intent to enforce all hub-to-hub routes go through Virtual WAN (ASN 65001) only":
        (True, True),

    # --- ECMP load balancing
    # Distributes traffic across equal-cost paths; does not change preferences or policy
    "Enable ECMP load balancing across hub1 and hub2 with equal-cost multipath":
        (False, False),

    # --- Disable hub peering entirely
    # Removes the BGP session → no route advertisements between hubs via peer
    # Cycle can't exist; leak path is eliminated
    "disable hub peering between vhubvnet1 and vhubvnet2":
        (True, True),

    # --- Wait for convergence / AS-PATH change
    # Passive; does not deterministically break cycle or stop leak
    "Wait for Virtual WAN automatic convergence or ASN 65003 to change AS_PATH length of routes":
        (False, False),

    # --- Restart BGP session on hub1
    # Resets routes temporarily; same misconfiguration reconverges → not resolved
    "Restart BGP session on hub1 ":
        (False, False),

    # --- Export policy blocking provider routes from hub1 to peer hub2
    # Directly stops hub1 from advertising VirtualWAN routes to hub2 → leak resolved
    # Hub2 loses the route via hub1 → hub2 preference for hub1 is broken → cycle broken
    "Configure export policy on vhubvnet1 (ASN 65002) to block announcing provider routes (from Virtual WAN ASN 65001) to peer vhubvnet2 (ASN 65003)":
        (True, True),

    # --- Route preference hierarchy: customer > WAN > peer on hub1
    # Hub1 prefers VirtualWAN routes over peer routes → hub1 no longer prefers hub2 path
    # Breaks hub1 side of the preference cycle
    # Does NOT stop hub1 from advertising to hub2
    "Set route preference hierarchy on hub1: customer routes (VNET1, VNET2) > Virtual WAN routes > peer routes (vhubvnet2)":
        (True, False),

    # --- Route dampening on hub1
    # Suppresses route flaps; does not fix preference cycle or stop advertisement
    "Enable route dampening on vhubvnet1 S2S VPN Gateway with penalty threshold 1000 and suppress limit 2000":
        (False, False),

    # --- Max-prefix limit between hub1 and hub2
    # May kill session if exceeded; does not fix policy or preferences
    "Set maximum-prefix limit of 10 prefixes on BGP session between vhubvnet1 and vhubvnet2":
        (False, False),

    # --- Export policy filtering routes learned FROM hub2 before re-advertising
    # Addresses a different direction (hub2→hub1→WAN), not the provider→peer leak
    # Does not change hub1's preference for hub2
    "Configure export policy on hub1 to filter out routes learned from hub2 before re-advertising":
        (False, False),

    # --- Deploy Route Map (generic / no specific policy defined)
    # Provides mechanism but no specific policy — does not resolve issues on its own
    "Deploy Route Map in hub1 with centralized routing management":
        (False, False),

    # --- Change BGP keepalive / holdtime timers
    # Timer tuning; does not affect routing logic, preferences, or policies
    "Change BGP keepalive timer of vhubvnet1 to 30 seconds and holdtime to 90 seconds":
        (False, False),

    # --- Update routing preference on hub1 to stop preferring routes via hub2
    # Directly removes hub1's preference for hub2 → breaks hub1 side of cycle
    # Does NOT stop hub1 from advertising provider routes to hub2
    "Update routing preference on hub1 (ASN 65002) to stop preferring routes via hub2 (ASN 65003)":
        (True, False),
}

# Build solution_results dict, stripping trailing whitespace from keys
solution_results = {}
for sol in possible_solutions:
    key = sol.strip()
    match_key = sol  # try exact first
    osc, leak = solution_verdicts.get(match_key, solution_verdicts.get(key, (False, False)))
    solution_results[sol] = {
        "oscillation_resolved": osc,
        "route_leak_resolved":  leak
    }

# =========================================================
# 4. ASSEMBLE AND WRITE REPORT
# =========================================================
report = {
    "oscillation_detected": oscillation_detected,
    "oscillation_cycle":    oscillation_cycle,
    "affected_ases":        affected_ases,
    "route_leak_detected":  route_leak_detected,
    "route_leaks":          route_leaks,
    "solution_results":     solution_results
}

import os; os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/oscillation_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport written to {out_path}")
print(json.dumps(report, indent=2))
