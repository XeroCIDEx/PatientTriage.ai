"""
surge_simulation.py
====================
A real, lightweight discrete-event queueing simulation of an ED under
normal vs. 3x surge volume — replacing the earlier static "count patients
twice" approximation.

WHAT IT MODELS (and what it deliberately does NOT):
  - Patients arrive over a simulated shift according to a Poisson
    process (random arrival times, average rate set by daily volume).
  - A FIXED number of "treatment slots" (e.g. nurses/bays) process
    patients — this number does NOT change between normal and surge
    mode, which is the whole point: surge stresses a fixed-capacity
    system.
  - Patients are pulled from the waiting queue by PRIORITY (lower ESI
    number = seen first), not strictly by arrival order — approximating
    real triage-queue behavior.
  - Each patient's simulated treatment duration depends on their ESI
    (critical patients occupy a slot longer) — an illustrative
    assumption, not derived from real ED throughput data.
  - We reuse MAX_SAFE_WAIT_MINUTES from monitoring.py to count how many
    patients in each mode would have BREACHED the safe-wait ceiling for
    their ESI level before being seen — this is the direct, quantified
    link between "more volume" and "more safety risk" that surge mode is
    meant to demonstrate.

WHAT THIS IS NOT: a validated hospital operations-research model. No
staffing schedule, no bed-turnover logic, no walk-outs, no
multi-resource contention (labs, imaging). It is a substantial upgrade
over a static multiplier, but still a simplified prototype simulation.
"""

from __future__ import annotations
import heapq
import numpy as np
from dataclasses import dataclass, field

from .monitoring import MAX_SAFE_WAIT_MINUTES

# Illustrative mean treatment-slot occupancy time (minutes) by ESI level.
MEAN_SERVICE_MINUTES = {1: 90, 2: 60, 3: 35, 4: 20, 5: 12}


@dataclass
class SimPatient:
    patient_id: str
    esi: int
    arrival_time: float
    wait_time: float = None
    service_start: float = None
    service_end: float = None
    safety_breach: bool = False


def run_surge_simulation(patients_per_day: int, esi_pool: list, n_servers: int = 5,
                          sim_minutes: int = 480, seed: int = 42) -> dict:
    """
    patients_per_day: target daily volume (used to derive average arrival rate)
    esi_pool: list of ESI levels (ints 1-5) to sample simulated arrivals from,
              e.g. the final_esi values already computed for the demo cohort —
              so the simulated mix reflects the same case distribution the
              rest of the prototype uses, not an arbitrary assumption.
    n_servers: fixed number of treatment slots available (same in both modes
               by design — see module docstring)
    sim_minutes: length of the simulated shift window (default 8 hours)
    """
    rng = np.random.default_rng(seed)
    # average inter-arrival time (minutes) to hit the target daily rate,
    # scaled down to the simulated window (assume the full day maps
    # proportionally onto sim_minutes for a self-contained shift demo)
    minutes_per_day = 24 * 60
    rate_per_minute = patients_per_day / minutes_per_day

    # generate arrival times via a Poisson process over sim_minutes
    arrivals = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate_per_minute)
        if t > sim_minutes:
            break
        esi = int(rng.choice(esi_pool))
        arrivals.append((t, esi))

    patients = [SimPatient(patient_id=f"SIM-{i:04d}", esi=esi, arrival_time=arr)
                for i, (arr, esi) in enumerate(arrivals)]

    # server free-times: n_servers slots, all free at t=0
    server_free_at = [0.0] * n_servers

    # Priority-queue discrete-event approximation: at each arrival, push
    # the new patient into a priority queue (lower ESI = more urgent =
    # served first, ties broken by arrival order), then greedily assign
    # any servers that are free by that point in time.
    events = sorted(patients, key=lambda p: p.arrival_time)
    queue = []
    next_free_idx = 0
    for p in events:
        heapq.heappush(queue, (p.esi, p.arrival_time, p))
        # try to assign as many free servers as possible at this arrival time
        now = p.arrival_time
        for i in range(n_servers):
            if server_free_at[i] <= now and queue:
                esi, arr, pat = heapq.heappop(queue)
                pat.service_start = max(now, server_free_at[i])
                pat.wait_time = pat.service_start - pat.arrival_time
                duration = max(2.0, rng.normal(MEAN_SERVICE_MINUTES[pat.esi], MEAN_SERVICE_MINUTES[pat.esi] * 0.25))
                pat.service_end = pat.service_start + duration
                server_free_at[i] = pat.service_end

    # anyone left in queue at end of arrivals: keep draining with soonest-free server
    while queue:
        esi, arr, pat = heapq.heappop(queue)
        i = int(np.argmin(server_free_at))
        pat.service_start = max(arr, server_free_at[i])
        pat.wait_time = pat.service_start - pat.arrival_time
        duration = max(2.0, rng.normal(MEAN_SERVICE_MINUTES[pat.esi], MEAN_SERVICE_MINUTES[pat.esi] * 0.25))
        pat.service_end = pat.service_start + duration
        server_free_at[i] = pat.service_end

    for p in patients:
        limit = MAX_SAFE_WAIT_MINUTES.get(p.esi, 120)
        p.safety_breach = (p.wait_time is not None) and (p.wait_time > limit)

    by_esi = {esi: [p.wait_time for p in patients if p.esi == esi] for esi in range(1, 6)}
    avg_wait_by_esi = {esi: (float(np.mean(v)) if v else None) for esi, v in by_esi.items()}
    breaches_by_esi = {esi: sum(1 for p in patients if p.esi == esi and p.safety_breach) for esi in range(1, 6)}

    return {
        "n_patients": len(patients),
        "n_servers": n_servers,
        "avg_wait_overall": float(np.mean([p.wait_time for p in patients])) if patients else 0.0,
        "max_wait_overall": float(np.max([p.wait_time for p in patients])) if patients else 0.0,
        "avg_wait_by_esi": avg_wait_by_esi,
        "safety_breaches_total": int(sum(breaches_by_esi.values())),
        "safety_breaches_by_esi": breaches_by_esi,
        "critical_count": sum(1 for p in patients if p.esi in (1, 2)),
        "urgent_count": sum(1 for p in patients if p.esi == 3),
        "non_urgent_count": sum(1 for p in patients if p.esi in (4, 5)),
        "patients": patients,
    }
