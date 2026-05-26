import json, os

DATA_DIR = "/workspace/results/app/data"
OUT_DIR  = "/workspace/results/app/output"

with open(f"{DATA_DIR}/route.json")              as f: route              = json.load(f)
with open(f"{DATA_DIR}/preferences.json")        as f: preferences        = json.load(f)
with open(f"{DATA_DIR}/local_pref.json")         as f: local_pref         = json.load(f)
with open(f"{DATA_DIR}/relationships.json")      as f: relationships      = json.load(f)
with open(f"{DATA_DIR}/topology.json")           as f: topology           = json.load(f)
with open(f"{DATA_DIR}/route_events.json")       as f: route_events       = json.load(f)
with open(f"{DATA_DIR}/possible_solutions.json") as f: possible_solutions = json.load(f)

# =========================================================
# 1. OSCILLATION DETECTION
#    Mutual routing-preference cycle in preferences.json
# =========================================================
pref_map = {int(k): int(v["prefer_via"]) for k, v in preferences.items()}
oscillation_cycle = []
checked = set()
for asn, via in pref_map.items():
    if asn not in checked and via in pref_map and pref_map[via] == asn:
        oscillation_cycle = [asn, via]
        checked.update([asn, via])
        break

oscillation_detected = bool(oscillation_cycle)
affected_ases = sorted(oscillation_cycle)

# =========================================================
# 2. ROUTE LEAK DETECTION
#    Valley-free: provider-learned routes MUST NOT be
#    re-advertised to peers or other providers.
#    Use the type fields already present in route_events.json.
# =========================================================
route_leaks = []
for ev in route_events:
    src_type = ev.get("source_type")
    dst_type = ev.get("destination_type")
    # Valley-free violation: provider route → peer  OR  provider route → provider
    if src_type == "provider" and dst_type in ("peer", "provider"):
        route_leaks.append({
            "leaker_as":       ev["advertiser_asn"],
            "source_as":       ev["source_asn"],
            "destination_as":  ev["destination_asn"],
            "source_type":     src_type,
            "destination_type": dst_type
        })

route_leak_detected = bool(route_leaks)

# =========================================================
# 3. SOLUTION EVALUATION
#    oscillation_resolved: must break the mutual preference
#      cycle between hub1 (65002) and hub2 (65003).
#    route_leak_resolved:  must stop hub1 from advertising
#      VirtualWAN (provider) routes to peer hub2 (egress).
# =========================================================
verdicts = {
    # Ingress prefix filter on hub1 for routes FROM hub2
    # → does not stop hub1's egress to hub2, does not change preferences
    "Apply route filter on vhubvnet1 to only accept prefixes 10.2.0.0/16 from vhubvnet2":
        (False, False),

    # no-export community on provider routes at hub1
    # → eBGP no-export stops hub1 from sending those routes to peer hub2
    # → hub2 loses the path via hub1 → hub2 side of cycle broken
    "Configure route policy on vhubvnet1 to enforce no-export of provider routes to peer vhubvnet2 by bgp community":
        (True, True),

    # hub1 prefers shorter AS-PATH: direct WAN path < hub2 path
    # → hub1 no longer prefers hub2 → cycle broken
    # → does NOT stop hub1 from advertising to hub2
    "Configure vhubvnet1 to prefer shorter AS-PATH length or use MED value 100 for route selection":
        (True, False),

    # RPKI validates origin AS; does not change routing preferences or egress policy
    "Enable RPKI origin validation on vhubvnet2 to verify prefix 10.2.1.0/24 originates from ASN 65007 (VNET4)":
        (False, False),

    # Ingress AS-PATH filter on hub2 rejects VirtualWAN routes received from hub1
    # → hub2 can no longer prefer hub1 for those routes → hub2 side of cycle broken
    # → hub1 still sends the routes (egress unchanged) → leak not stopped
    "Enable ingress filtering on vhubvnet2 (ASN 65003) to reject routes with AS_PATH containing Virtual WAN ASN 65001 when received from peer vhubvnet1":
        (True, False),

    # UDR override forces hub1 to use a specific next-hop (not hub2)
    # → breaks hub1's preference for hub2 → cycle broken
    # → does not change hub1's BGP advertisement policy toward hub2
    "Configure user defined route override on vhubvnet1 defaultRouteTable to prefer specific next-hop":
        (True, False),

    # Hub Routing Intent forces all hub-to-hub traffic through VirtualWAN
    # → effectively removes direct-peer route advertisements → cycle broken, leak stopped
    "Enable Virtual WAN hub routing intent to enforce all hub-to-hub routes go through Virtual WAN (ASN 65001) only":
        (True, True),

    # ECMP distributes load; does not change preferences or advertisement policy
    "Enable ECMP load balancing across hub1 and hub2 with equal-cost multipath":
        (False, False),

    # Removing the BGP peering session eliminates the route-leak channel and the
    # preference cycle entirely
    "disable hub peering between vhubvnet1 and vhubvnet2":
        (True, True),

    # Passive wait; no deterministic policy change
    "Wait for Virtual WAN automatic convergence or ASN 65003 to change AS_PATH length of routes":
        (False, False),

    # Restarting BGP resets routes; same misconfiguration re-converges after restart
    "Restart BGP session on hub1 ":
        (False, False),

    # Export policy explicitly blocks hub1 from announcing VirtualWAN routes to peer hub2
    # → directly stops leak; hub2 loses the path via hub1 → cycle broken
    "Configure export policy on vhubvnet1 (ASN 65002) to block announcing provider routes (from Virtual WAN ASN 65001) to peer vhubvnet2 (ASN 65003)":
        (True, True),

    # Route preference hierarchy: WAN > peer on hub1
    # → hub1 prefers WAN route over hub2 route → hub1 side of cycle broken
    # → does not stop hub1 from advertising to hub2
    "Set route preference hierarchy on hub1: customer routes (VNET1, VNET2) > Virtual WAN routes > peer routes (vhubvnet2)":
        (True, False),

    # Route dampening suppresses flaps; does not fix root cause
    "Enable route dampening on vhubvnet1 S2S VPN Gateway with penalty threshold 1000 and suppress limit 2000":
        (False, False),

    # Max-prefix limit may tear down session but is not a routing-policy fix
    "Set maximum-prefix limit of 10 prefixes on BGP session between vhubvnet1 and vhubvnet2":
        (False, False),

    # Filters routes FROM hub2 before re-advertising; does not address provider→peer leak
    "Configure export policy on hub1 to filter out routes learned from hub2 before re-advertising":
        (False, False),

    # Generic Route Map deployment without specific policy defined → no concrete resolution
    "Deploy Route Map in hub1 with centralized routing management":
        (False, False),

    # Timer tuning only; no routing logic changed
    "Change BGP keepalive timer of vhubvnet1 to 30 seconds and holdtime to 90 seconds":
        (False, False),

    # Directly removes hub1's preference for hub2 → cycle broken
    # → does not stop hub1 from advertising provider routes to hub2
    "Update routing preference on hub1 (ASN 65002) to stop preferring routes via hub2 (ASN 65003)":
        (True, False),
}

solution_results = {}
for sol in possible_solutions:
    key = sol.strip()
    osc, leak = verdicts.get(sol, verdicts.get(key, (False, False)))
    solution_results[sol] = {
        "oscillation_resolved": osc,
        "route_leak_resolved":  leak
    }

# =========================================================
# 4. WRITE REPORT
# =========================================================
report = {
    "oscillation_detected": oscillation_detected,
    "oscillation_cycle":    oscillation_cycle,
    "affected_ases":        affected_ases,
    "route_leak_detected":  route_leak_detected,
    "route_leaks":          route_leaks,
    "solution_results":     solution_results
}

os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/oscillation_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Written: {out_path}")
print(json.dumps(report, indent=2))
