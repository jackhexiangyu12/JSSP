"""
JSSP Solver: LP Relaxation + Heuristic Rounding
Author: Wenbin Zhai
Date: 2025-10-17

This script solves the Job-Shop Scheduling Problem (JSSP) using a method
of LP relaxation followed by heuristic rounding.
It utilizes the 'job-shop-lib' library to load benchmark instances and to
visualize the results, incorporating advanced visualization techniques
from the library's documentation, including static and dynamic Gantt charts.
All generated plots are saved to files instead of being displayed.

Core Dependencies:
- pulp: For building and solving the LP model.
- networkx: For constructing the schedule graph and calculating the longest path.
- job-shop-lib: For loading benchmark instance data and visualizing results.
- matplotlib: For displaying the Gantt chart.
- imageio: For creating the animated GIF from plot frames.
- pygraphviz (optional but recommended): For better graph layouts.
"""

import time
from collections import defaultdict
import pulp
import networkx as nx
# import matplotlib.pyplot as plt
# import os
# import shutil
# import imageio
# import matplotlib.patches as mpatches
# import argparse

from job_shop_lib.benchmarking import load_benchmark_instance


# from job_shop_lib import Schedule, ScheduledOperation
# from job_shop_lib.visualization.gantt import plot_gantt_chart
# from job_shop_lib.visualization.graphs import plot_disjunctive_graph, plot_resource_task_graph
# from job_shop_lib.graphs import build_solved_disjunctive_graph, build_resource_task_graph


def solve_jssp_with_lp_relaxation_and_rounding(instance):
    """
    Solves a JSSP instance using LP relaxation and heuristic rounding.
    """
    num_jobs = instance.num_jobs
    num_machines = instance.num_machines
    operations = instance.jobs

    # --- Phase 1: Build and solve the LP relaxation model ---
    print("--- Phase 1: Building and solving LP relaxation model... ---")
    start_time_lp = time.time()

    model = pulp.LpProblem("JSSP_LP_Relaxation", pulp.LpMinimize)

    # Variable definitions
    s = pulp.LpVariable.dicts("s", ((i, j) for i in range(num_jobs) for j in range(num_machines)), lowBound=0)
    C_max = pulp.LpVariable("C_max", lowBound=0)

    # Binary variables for disjunctive constraints (relaxed to continuous)
    y = pulp.LpVariable.dicts("y", ((m, i1, j1, i2, j2)
                                    for m in range(num_machines)
                                    for i1 in range(num_jobs) for j1 in range(num_machines)
                                    for i2 in range(num_jobs) for j2 in range(num_machines)
                                    if (i1, j1) < (i2, j2)),
                              lowBound=0, upBound=1)

    # Objective function
    model += C_max

    # A very large positive number
    BIG_M = sum(op.duration for job in operations for op in job)

    # Constraints
    ops_on_machine = defaultdict(list)
    for i in range(num_jobs):
        for j in range(num_machines):
            op_obj = operations[i][j]
            machine = op_obj.machines[0]
            ops_on_machine[machine].append((i, j))

    for i in range(num_jobs):
        for j in range(num_machines):
            op_obj = operations[i][j]
            p_ij = op_obj.duration

            # Precedence constraints
            if j > 0:
                prev_op_obj = operations[i][j - 1]
                p_prev = prev_op_obj.duration
                model += s[i, j] >= s[i, j - 1] + p_prev

            # C_max constraint
            model += C_max >= s[i, j] + p_ij

    # Disjunctive constraints
    for m in range(num_machines):
        ops = ops_on_machine[m]
        for k1 in range(len(ops)):
            for k2 in range(k1 + 1, len(ops)):
                i1, j1 = ops[k1]
                i2, j2 = ops[k2]

                p1 = operations[i1][j1].duration
                p2 = operations[i2][j2].duration

                # Ensure the key order for y is fixed
                y_key_tuple = (m, i1, j1, i2, j2)

                model += s[i1, j1] >= s[i2, j2] + p2 - BIG_M * y[y_key_tuple]
                model += s[i2, j2] >= s[i1, j1] + p1 - BIG_M * (1 - y[y_key_tuple])

    # Solve the LP model
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    lp_lower_bound = pulp.value(model.objective)
    end_time_lp = time.time()
    print(f"LP relaxation solved. Time: {end_time_lp - start_time_lp:.2f} seconds.")
    print(f"LP objective (Makespan lower bound): {lp_lower_bound:.2f}")

    # --- Phase 2: Heuristic rounding and final schedule construction ---
    print("--- Phase 2: Heuristic rounding and building final schedule... ---")
    start_time_heuristic = time.time()

    # Extract start times from the LP solution as heuristic information
    lp_start_times = {}
    for i in range(num_jobs):
        for j in range(num_machines):
            if s[i, j].varValue is not None:
                lp_start_times[(i, j)] = s[i, j].varValue
            else:
                lp_start_times[(i, j)] = 0  # Default if None

    # Build a Directed Acyclic Graph (DAG)
    G = nx.DiGraph()
    SOURCE, SINK = 'SOURCE', 'SINK'

    # Add edges for precedence constraints
    for i in range(num_jobs):
        G.add_edge(SOURCE, (i, 0), weight=0)
        for j in range(num_machines - 1):
            duration = operations[i][j].duration
            G.add_edge((i, j), (i, j + 1), weight=duration)
        last_op_duration = operations[i][num_machines - 1].duration
        G.add_edge((i, num_machines - 1), SINK, weight=last_op_duration)

    # Add edges for machine disjunctive constraints (heuristic ordering based on LP solution)
    for m in range(num_machines):
        sorted_ops = sorted(ops_on_machine[m], key=lambda op: lp_start_times.get(op, 0))
        for k in range(len(sorted_ops) - 1):
            op1_i, op1_j = sorted_ops[k]
            op2_i, op2_j = sorted_ops[k + 1]
            duration1 = operations[op1_i][op1_j].duration
            G.add_edge((op1_i, op1_j), (op2_i, op2_j), weight=duration1)

    for u, v, d in G.edges(data=True):
        d['weight'] = -d['weight']

    shortest_path_lengths = nx.single_source_bellman_ford_path_length(G, source=SOURCE, weight='weight')
    final_schedule_start_times = {node: -length for node, length in shortest_path_lengths.items()}
    final_makespan = final_schedule_start_times.get(SINK, 0)

    end_time_heuristic = time.time()
    print(f"Heuristic schedule constructed. Time: {end_time_heuristic - start_time_heuristic:.2f} seconds.")
    return final_makespan, lp_lower_bound, final_schedule_start_times


# def plot_instance_graph(instance, instance_name, schedule: "Schedule" = None):
#     """Saves the disjunctive graph to a file."""
#     print(f"--- Generating and saving Disjunctive Graph for '{instance_name}'... ---")
#     if schedule is not None:
#         try:
#             solved_graph = build_solved_disjunctive_graph(schedule)
#             fig, ax = plot_disjunctive_graph(
#                 solved_graph,
#                 figsize=(12, 8),
#                 draw_disjunctive_edges=True,
#                 disjunctive_edges_additional_params={"arrowstyle": "-|>", "connectionstyle": "arc3,rad=0.12"},
#                 conjunctive_edges_additional_params={"arrowstyle": "-|>", "connectionstyle": "arc3,rad=0.0"},
#             )
#             title = f"Disjunctive Graph (Oriented by Solution) for '{instance_name}'"
#             filename = f"{instance_name}_disjunctive_graph_solved.png"
#             # fig.suptitle(title, fontsize=16)
#             plt.savefig(filename, bbox_inches='tight')
#             plt.close(fig)
#             print(f"  -> Saved solved disjunctive graph to '{filename}'")
#         except Exception as e:
#             print(f"Warning: could not build solved disjunctive graph: {e}")
#     else:
#         fig, ax = plot_disjunctive_graph(
#             instance,
#             figsize=(10, 6),
#             draw_disjunctive_edges="single_edge",
#             disjunctive_edges_additional_params={"arrowstyle": "<->", "connectionstyle": "arc3,rad=0.12"},
#         )
#         title = f"Disjunctive Graph (Problem Structure) for '{instance_name}'"
#         filename = f"{instance_name}_disjunctive_graph_initial.png"
#         # fig.suptitle(title, fontsize=16)
#         plt.savefig(filename, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  -> Saved initial disjunctive graph to '{filename}'")
#
#
# def plot_resource_task_graph_visualization(instance, instance_name):
#     """Saves the resource-task graph to a file."""
#     print(f"--- Generating and saving Resource-Task Graph for instance '{instance_name}'... ---")
#     try:
#         resource_task_graph = build_resource_task_graph(instance)
#         fig = plot_resource_task_graph(
#             resource_task_graph,
#             figsize=(12, 10)
#         )
#         title = f"Resource-Task Graph for '{instance_name}'"
#         filename = f"{instance_name}_resource_task_graph.png"
#         fig.suptitle(title, fontsize=16)
#         plt.savefig(filename, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  -> Saved resource-task graph to '{filename}'")
#     except Exception as e:
#         print(f"Failed to generate Resource-Task Graph: {e}")
#
#
# def create_and_save_gantt(instance, final_start_times, instance_name, makespan, color_mapper):
#     """Saves the Gantt chart, manually recoloring bars to ensure consistency."""
#     print(f"--- Generating and saving Gantt chart for instance '{instance_name}'... ---")
#     scheduled_operations = []
#     for job_idx, job in enumerate(instance.jobs):
#         for op_idx, operation in enumerate(job):
#             node_name = (job_idx, op_idx)
#             start_time = final_start_times.get(node_name)
#             if start_time is not None:
#                 machine_id = operation.machines[0]
#                 so = ScheduledOperation(
#                     operation=operation, start_time=float(start_time), machine_id=int(machine_id)
#                 )
#                 scheduled_operations.append(so)
#
#     print(f"  -> Total scheduled operations collected: {len(scheduled_operations)}")
#     schedule_by_machine = [[] for _ in range(instance.num_machines)]
#     for so in scheduled_operations:
#         schedule_by_machine[so.machine_id].append(so)
#     for m_ops in schedule_by_machine:
#         m_ops.sort(key=lambda op: op.start_time)
#
#     schedule = Schedule(instance=instance, schedule=schedule_by_machine)
#     fig, ax = plot_gantt_chart(
#         schedule,
#         title=f"Gantt Chart for '{instance_name}' (Makespan: {makespan:.0f})",
#         xlim=makespan,
#     )
#
#     for container in ax.containers:
#         label = container.get_label()
#         if label and label.startswith('Job'):
#             try:
#                 job_id = int(label.split()[-1])
#                 desired_color = color_mapper.get(job_id)
#                 if desired_color:
#                     container.set_color(desired_color)
#             except (ValueError, IndexError):
#                 continue  # Ignore labels that don't parse to a job ID
#
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:
#         ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
#     if makespan and makespan > 0:
#         ax.set_xlim(0, makespan * 1.05)
#
#     filename = f"{instance_name}_gantt_chart.png"
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  -> Saved Gantt chart to '{filename}'")
#     return schedule
#
#
# def generate_gantt_animation_from_schedule(schedule, gif_path, instance_name, makespan, color_mapper, fps=2):
#     """Creates a GIF, manually recoloring bars in each frame for consistency."""
#     print(f"--- Generating animated Gantt chart GIF for '{instance_name}'... ---")
#     frame_dir = "temp_gantt_frames"
#     if os.path.exists(frame_dir):
#         shutil.rmtree(frame_dir)
#     os.makedirs(frame_dir)
#
#     all_ops = sorted([op for machine_ops in schedule.schedule for op in machine_ops], key=lambda op: op.start_time)
#     filenames = []
#
#     for i in range(len(all_ops)):
#         partial_ops = all_ops[:i + 1]
#         schedule_by_machine = [[] for _ in range(schedule.instance.num_machines)]
#         for so in partial_ops:
#             schedule_by_machine[so.machine_id].append(so)
#         partial_schedule = Schedule(instance=schedule.instance, schedule=schedule_by_machine)
#
#         current_time_for_frame = all_ops[i].end_time
#
#         fig, ax = plot_gantt_chart(
#             partial_schedule,
#             title=f"Gantt Animation for '{instance_name}' (Operation {i + 1}/{len(all_ops)})",
#             xlim=makespan,
#         )
#
#         ax.axvline(x=current_time_for_frame, color='red', linestyle='--', linewidth=1.5)
#
#         for container in ax.containers:
#             label = container.get_label()
#             if label and label.startswith('Job'):
#                 try:
#                     job_id = int(label.split()[-1])
#                     desired_color = color_mapper.get(job_id)
#                     if desired_color:
#                         container.set_color(desired_color)
#                 except (ValueError, IndexError):
#                     continue
#
#         handles, labels = ax.get_legend_handles_labels()
#         if handles:
#             ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
#         if makespan and makespan > 0:
#             ax.set_xlim(0, makespan * 1.05)
#
#         filename = f"{frame_dir}/frame_{i:04d}.png"
#         filenames.append(filename)
#         plt.savefig(filename, bbox_inches='tight')
#         plt.close(fig)
#
#     with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
#         for filename in filenames:
#             try:
#                 image = imageio.v2.imread(filename)
#                 writer.append_data(image)
#             except Exception as e:
#                 print(f"Warning: Could not process frame {filename}: {e}")
#
#     shutil.rmtree(frame_dir)
#     print(f"Successfully saved animation to '{gif_path}'")


def main():
    # parser = argparse.ArgumentParser(description="JSSP Solver using LP Relaxation and Heuristic Rounding.")
    # parser.add_argument('datasets', nargs='+', help="One or more JSSP benchmark instance names to solve (e.g., ft06 ft10 la02).")
    # args = parser.parse_args()

    print("=" * 60)
    print("  JSSP Solver: LP Relaxation + Heuristic Rounding")
    print("=" * 60)

    # --- Tunable Parameters ---
    # Generate instance names la01, la02, ..., la40
    INSTANCES_TO_SOLVE = [f"la{i:02d}" for i in range(1, 41)]

    # Disable all visualization
    SAVE_GANTT_CHART = False
    SAVE_DISJUNCTIVE_GRAPH = False
    SAVE_RESOURCE_TASK_GRAPH = False
    SAVE_GANTT_CHART_GIF = False

    # plt.style.use("ggplot")
    results = []

    for instance_name in INSTANCES_TO_SOLVE:
        print(f"\n\n{'=' * 20} Solving instance: {instance_name.upper()} {'=' * 20}")
        optimal_makespan = None  # Initialize

        try:
            print(f"Loading instance from job-shop-lib: {instance_name} ...")
            instance = load_benchmark_instance(instance_name)

            # --- CORRECTED LINE ---
            # Get the known optimal makespan from the instance's metadata
            optimal_makespan = instance.metadata.get("optimum")
            # ----------------------

            if optimal_makespan is None:
                print(f"Warning: No 'optimum' value found in metadata for '{instance_name}'.")

            print(
                f"Instance loaded successfully: {instance.num_jobs} jobs, {instance.num_machines} machines. (Known Optimum: {optimal_makespan})")

        except Exception as e:
            print(f"Failed to load or process instance '{instance_name}': {e}")
            continue

        # num_jobs = instance.num_jobs
        # colors = plt.get_cmap('tab10', num_jobs)
        # color_mapper = {i: colors(i) for i in range(num_jobs)}

        # if SAVE_RESOURCE_TASK_GRAPH:
        #     plot_resource_task_graph_visualization(instance, instance_name)
        # if SAVE_DISJUNCTIVE_GRAPH:
        #     plot_instance_graph(instance, instance_name)

        makespan, lower_bound, final_schedule_times = solve_jssp_with_lp_relaxation_and_rounding(instance)

        # Store results including the optimal makespan
        results.append({
            "instance": instance_name,
            "makespan": makespan,
            "lower_bound": lower_bound,
            "optimal": optimal_makespan
        })
        print(f"\n>>> Instance '{instance_name}' solved. Final Makespan: {makespan:.2f}")

        # schedule_object = None
        # if (SAVE_GANTT_CHART or SAVE_DISJUNCTIVE_GRAPH or SAVE_GANTT_CHART_GIF) and final_schedule_times:
        #     schedule_object = create_and_save_gantt(
        #         instance, final_schedule_times, instance_name, makespan, color_mapper
        #     )
        #
        # if SAVE_DISJUNCTIVE_GRAPH and schedule_object:
        #     plot_instance_graph(instance, instance_name, schedule=schedule_object)
        #
        # if SAVE_GANTT_CHART_GIF and schedule_object:
        #     gif_path = f"{instance_name}_solution_animation.gif"
        #     generate_gantt_animation_from_schedule(
        #         schedule=schedule_object,
        #         gif_path=gif_path,
        #         instance_name=instance_name,
        #         makespan=makespan,
        #         color_mapper=color_mapper,
        #         fps=10
        #     )

    print("\n\n" + "=" * 74)
    print(" " * 27 + "Final Results Summary")
    print("-" * 74)
    print(f"{'Instance Name':<15} | {'Makespan (Heuristic)':<20} | {'LP Lower Bound':<15} | {'Optimal Makespan':<20}")
    print("-" * 74)
    for res in results:
        # Handle None optimal makespan for cleaner printing
        optimal_str = f"{res['optimal']}" if res['optimal'] is not None else "N/A"
        print(f"{res['instance']:<15} | {res['makespan']:<20.2f} | {res['lower_bound']:<15.2f} | {optimal_str:<20}")
    print("-" * 74)
    print("\nAll specified tasks have been completed.")


if __name__ == "__main__":
    main()