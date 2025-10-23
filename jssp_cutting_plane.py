# jssp_la11_cutting_plane.py
# Cutting-plane method for LA11 benchmark
# pip install pulp job-shop-lib matplotlib

import itertools
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib import Operation
import pulp
import matplotlib.pyplot as plt
import numpy as np


def horizon_estimate(ops):
    """Estimate time horizon for Big-M constraints"""
    total_processing = sum(o['proc'] for o in ops)
    return int(total_processing * 1.5)  # Conservative estimate for LA11

def build_disjunctive_milp(ops, horizon=None, enforce_binary=True):
    """Build MILP model for JSSP using disjunctive programming"""
    model = pulp.LpProblem("JSSP_LA11", pulp.LpMinimize)
    S = {}  # Start time variables
    for o in ops:
        S[o['id']] = pulp.LpVariable(f"S_{o['id'][0]}_{o['id'][1]}", lowBound=0, cat='Continuous')

    Cmax = pulp.LpVariable("Cmax", lowBound=0, cat='Continuous')
    model += Cmax

    # Precedence constraints within each job
    jobs = {}
    for o in ops:
        jobs.setdefault(o['job'], []).append(o)

    for j, job_ops in jobs.items():
        job_ops_sorted = sorted(job_ops, key=lambda x: x['idx'])
        for a, b in zip(job_ops_sorted[:-1], job_ops_sorted[1:]):
            model += S[b['id']] >= S[a['id']] + a['proc']

    # Disjunctive constraints for operations on same machine
    if horizon is None:
        horizon = horizon_estimate(ops)
    M = horizon

    X = {}
    machine_ops = {}

    # Group operations by machine first for efficiency
    for o in ops:
        machine_ops.setdefault(o['mach'], []).append(o)

    for mach, mach_ops_list in machine_ops.items():
        for i, o1 in enumerate(mach_ops_list):
            for j, o2 in enumerate(mach_ops_list):
                if i >= j:  # Avoid duplicate pairs
                    continue
                a = o1['id']; b = o2['id']
                if enforce_binary:
                    X[(a,b)] = pulp.LpVariable(f"x_{a[0]}_{a[1]}_{b[0]}_{b[1]}", cat='Binary')
                else:
                    X[(a,b)] = pulp.LpVariable(f"x_{a[0]}_{a[1]}_{b[0]}_{b[1]}", lowBound=0, upBound=1, cat='Continuous')

                # Disjunctive constraints using Big-M
                model += S[a] + o1['proc'] <= S[b] + M * (1 - X[(a,b)])
                model += S[b] + o2['proc'] <= S[a] + M * X[(a,b)]

    # Makespan constraints
    for o in ops:
        model += Cmax >= S[o['id']] + o['proc']

    return model, S, X, Cmax

def solve_pulp_model(model, time_limit=None, msg=True):
    """Solve the model with time limit"""
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    model.solve(solver)
    return pulp.LpStatus[model.status], pulp.value(model.objective)

def extract_schedule(S_vars):
    """Extract schedule from solved variables"""
    return {k: pulp.value(v) for k, v in S_vars.items() if pulp.value(v) is not None}

def compute_makespan(schedule, ops):
    """Compute makespan from schedule"""
    return max(schedule.get(o['id'], 0) + o['proc'] for o in ops)

def operations_overlap(start_a, duration_a, start_b, duration_b, tolerance=1e-4):
    """Check if two operations overlap in time"""
    return (start_a < start_b + duration_b - tolerance and
            start_b < start_a + duration_a - tolerance)

def get_precedence_decision(a, b, o1, o2, X_vars, Svals):
    """Decide operation precedence based on LP solution"""
    x_val = None
    for key in [(a, b), (b, a)]:
        if key in X_vars:
            x_val = pulp.value(X_vars[key])
            break

    if x_val is None:
        return None

    sa, sb = Svals.get(a, 0), Svals.get(b, 0)

    # Decision logic based on confidence and time difference
    if x_val > 0.9:  # High confidence: a before b
        return 'a_before_b'
    elif x_val < 0.1:  # High confidence: b before a
        return 'b_before_a'
    # elif abs(sa - sb) > max(o1['proc'], o2['proc']) / 2:
    #     # Clear time difference
    #     return 'a_before_b' if sa < sb else 'b_before_a'

    return None  # Not confident enough

# -------------------------
# Cutting-plane method for LA
# -------------------------
def run_cutting_plane_la(ops, horizon=None, max_iters=100, lp_time_limit=60, milp_time_limit=300):
    """
    Cutting-plane method specifically tuned for LA
    """

    # Phase 1: Initialize with LP relaxation
    print("Building initial LP relaxation...")
    model, S, X, Cmax = build_disjunctive_milp(ops, horizon, enforce_binary=False)

    # Store cut information (not constraint objects)
    accumulated_cut_info = []  # List of (a, b, order) tuples
    best_lp_obj = float('inf')
    iteration_stats = []

    # Phase 2: Cutting-plane iterations
    for iteration in range(max_iters):

        # Solve LP relaxation
        status, obj = solve_pulp_model(model, time_limit=lp_time_limit, msg=False)

        if status != 'Optimal':
            # print(f"LP solve failed with status: {status}")
            break

        Svals = extract_schedule(S)
        # print(f"LP Objective (Cmax): {obj:.2f}")

        # Track best LP solution
        if obj < best_lp_obj:
            best_lp_obj = obj

        # Detect and add cuts for overlapping operations
        cuts_added = 0
        machine_ops = {}

        # Group operations by machine
        for o in ops:
            machine_ops.setdefault(o['mach'], []).append(o)

        for mach, mach_ops_list in machine_ops.items():
            # Check all pairs on this machine
            for i in range(len(mach_ops_list)):
                for j in range(i + 1, len(mach_ops_list)):
                    o1 = mach_ops_list[i]
                    o2 = mach_ops_list[j]
                    a, b = o1['id'], o2['id']

                    sa = Svals.get(a, 0)
                    sb = Svals.get(b, 0)

                    # Check for overlap
                    if operations_overlap(sa, o1['proc'], sb, o2['proc']):
                        # Get precedence decision
                        order = get_precedence_decision(a, b, o1, o2, X, Svals)

                        if order is not None:
                            # Add cut to current model
                            if order == 'a_before_b':
                                model += S[a] + o1['proc'] <= S[b]
                                accumulated_cut_info.append((a, b, 'a_before_b'))
                                cuts_added += 1
                            else:  # 'b_before_a'
                                model += S[b] + o2['proc'] <= S[a]
                                accumulated_cut_info.append((a, b, 'b_before_a'))
                                cuts_added += 1

        iteration_stats.append({
            'iteration': iteration + 1,
            'lp_objective': obj,
            'cuts_added': cuts_added,
            'total_cuts': len(accumulated_cut_info)
        })

        # Stopping criteria
        if cuts_added == 0:
            print("No overlapping pairs detected. Solution is feasible.")
            break

        # Early stopping if objective deteriorates significantly
        if iteration >= 5 and obj > best_lp_obj + 30:
            print("Objective deteriorated significantly, stopping early")
            break

    final_model, final_S, final_X, final_Cmax = build_disjunctive_milp(ops, horizon, enforce_binary=True)

    # Add all accumulated precedence constraints
    strong_cuts = 0
    for a, b, order in accumulated_cut_info:
        o1 = next(o for o in ops if o['id'] == a)
        o2 = next(o for o in ops if o['id'] == b)

        if order == 'a_before_b':
            final_model += final_S[a] + o1['proc'] <= final_S[b]
        else:
            final_model += final_S[b] + o2['proc'] <= final_S[a]
        strong_cuts += 1

    print(f"Added {strong_cuts} strong precedence constraints to final MILP")


    # Solve final MILP
    print("Solving final MILP...")
    final_status, final_obj = solve_pulp_model(final_model, time_limit=milp_time_limit, msg=True)
    final_schedule = extract_schedule(final_S)

    return final_status, final_obj, final_schedule, iteration_stats

def add_strong_precedence_from_lp(final_model, final_S, ops, best_lp_solution):
    """Add strong precedence constraints based on LP solution"""
    if best_lp_solution is None:
        return 0

    cuts_added = 0
    machine_ops = {}

    for o in ops:
        machine_ops.setdefault(o['mach'], []).append(o)

    for mach, mach_ops_list in machine_ops.items():
        # Sort operations by start time in LP solution
        sorted_ops = sorted(mach_ops_list, key=lambda o: best_lp_solution.get(o['id'], 0))

        # Add precedence constraints for consecutive operations with clear ordering
        for i in range(len(sorted_ops) - 1):
            o1 = sorted_ops[i]
            o2 = sorted_ops[i + 1]

            time_gap = best_lp_solution.get(o2['id'], 0) - best_lp_solution.get(o1['id'], 0)
            if time_gap > o1['proc']:  # Significant gap
                final_model += final_S[o1['id']] + o1['proc'] <= final_S[o2['id']]
                cuts_added += 1

    return cuts_added




# -------------------------
# Main execution for LA11
# -------------------------
def main():

    datasets = {f"LA{i:02d}": None for i in range(1, 41)}

    results = {}
    for name in datasets.keys():

      inst = load_benchmark_instance(name.lower())
      # Convert to internal format
      ops = []
      for j, job in enumerate(inst.jobs):
          for k, operation in enumerate(job):
              ops.append({
                  'id': (j, k),
                  'job': j,
                  'idx': k,
                  'mach': operation.machine_id,
                  'proc': operation.duration
              })

      horizon = horizon_estimate(ops)

      # Run cutting-plane method with tuned parameters for LA
      status, obj, schedule, stats = run_cutting_plane_la(
          ops,
          horizon=horizon,
          max_iters=100,
          lp_time_limit=60,    # Shorter for smaller problem
          milp_time_limit=180  # 3 minutes for final MILP
      )
      results[name] = obj
      print(f"name:[{name}], status:[{status}], makespan:[{obj}]")

      print("results", results)





if __name__ == "__main__":
    main()