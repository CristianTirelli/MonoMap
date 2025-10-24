#!/usr/bin/env python3

import sys
import math
import time
import random
import itertools
import xml.dom.minidom
import threading
import argparse
import csv
import pickle

from random import randint

import xml.etree.ElementTree as ET
import networkx as nx

from z3 import *
from queue import Queue, Empty
from networkx.drawing.nx_agraph import write_dot
from networkx.algorithms import isomorphism
from collections import defaultdict




def getMaxOutDegree(dfg):

    max_degree = 0
    for n in dfg.nodes:
        max_degree = max(max_degree, dfg.out_degree(n))
    
    return max_degree


def check_solution(pe_nodes, dfg, size_y, size_x):

    for e in dfg.edges:
        source = e[0]
        destination = e[1]
        for ps in pe_nodes:
            if source in pe_nodes[ps]:
                for pd in pe_nodes:
                    if destination in pe_nodes[pd]:
                        if isConnected(pd,ps,size_y, size_x) == False:
                            return False
    return True

def getMobilityValue(kms, node):
    mobility = 0
    for t in kms:
        for p in kms[t]:
            if p[1] == node:
                mobility += 1
    return mobility

def map(dfg, arch, II, topology_degree, size_y, size_x):
    array_size = size_y * size_x
    # TODO: should put in the que the mapping
    total_time = 0
    # Generate valid schedule
    print("Start schedule generation")
    start = time.time()
    schedule = generate_valid_schdule(dfg, II, array_size, topology_degree)
    end = time.time()
    print("End schedule generation: ", (end - start))

    schedule[0] = [0, 2, 5]
    schedule[1] = [1, 3, 6]
    schedule[2] = [4]

    total_time += (end-start)
    II = len(schedule)
    print("Len schedule", II)
    print("Schedule")
    for i in range(0, II):
        for n in dfg.nodes:
            count = 0
            for succ in list(dfg.successors(n)):
                if int(succ) in schedule[i]:
                    count += 1
            if count > 5:
                print("Node ", n, "overscheduling of childs")
                exit(0)
        print(i, sorted(schedule[i]))
    #II = 70
    #schedule[0] = [ 16, 3, 25, 40, 47, 4, 10, 29, 34, 0, 20, 46, 43, 28, 7, 50 ,11, 38 ]
    #schedule[1] = [ 30, 21, 39, 49, 37, 44, 26, 41, 5, 23, 17, 19, 32, 1, 8, 14, 35, 12 ]
    #schedule[2] = [ 42, 13, 33, 24, 22, 48, 6, 27, 15, 36, 9, 18, 2, 45, 31 ]       
    # Generate architecture graph

    # Generate nodes for each time step
    #node_id = 0
    #t = II
    #nodes = {}
    #for i in range(t):
    #    if i not in nodes:
    #        nodes[i] = []
    #    for m in range(0, size_x):
    #        for n in range(0, size_y):
    #            nodes[i].append(node_id)
    #            node_id += 1
#
    #t = II
    ## Add dependency edges
    #print("Start architecture graph generation")
    #start = time.time()
    #for i in range(0, t):
    #    for j in range(0, t):
    #        #if i == j: continue
    #        for n_i in nodes[i]:
    #            for n_j in nodes[j]:
    #                if n_i == n_j and i==j: 
    #                    continue
#
    #                if isConnected(n_i % (size_x * size_y), n_j % (size_x * size_y), size_y, size_x):
    #                    arch.add_node(n_i)
    #                    arch.add_node(n_j)
#
    #                    arch.nodes[n_i]['time'] = i
    #                    arch.nodes[n_j]['time'] = j
    #
    #                    arch.add_edge(n_i, n_j)
    #end = time.time()
    print("Time to generate architecture: " + str(end - start))                  
    #load graph 
    with open("arch_graph20x20_20.pkl", "rb") as f:
        arch = pickle.load(f)


    #write_dot(G1, "G1.dot")
    # Convert DFG to undirected graph
    dfg = dfg.to_undirected()

    print("Monomorphism search start...")

    # Assign attributes to DFG nodes
    # the attribute for every node is 
    # the time steps at which the node is scheduled
    for n in dfg.nodes:
        dfg.nodes[n]['time'] = getTime(int(n), schedule)

    node_pe = {}
    pe_nodes = {}
    # Start monomorphism search
    nm = isomorphism.categorical_node_match("time", [i for i in range(II)])
    #GM = nx.isomorphism.GraphMatcher(arch, dfg)
    GM = nx.isomorphism.GraphMatcher(arch, dfg, nm)
    start = time.time()
    ii = 0
    row = []
    new_row = [""]
    for t in range(II):
        for p in range(size_x * size_y):
            new_row.append("PE" + str(p) + " at time " + str(t))
    new_row.append("Valid")
    row.append(new_row)
    

    for m in GM.subgraph_monomorphisms_iter():
        #print(m)
        node_pe = {}
        pe_nodes = {}
        for k in m:
            #print("Node: " + str(m[k]) + " on PE: " + str(k) + "=" + str(k % (size_x*size_y)) )
            if k % (size_x * size_y) not in pe_nodes:
                pe_nodes[k % (size_x * size_y)] = []
            pe_nodes[k % (size_x * size_y)].append(m[k])

            if m[k] not in node_pe:
                node_pe[m[k]] = k % (size_x * size_y)
            else:
                print("should not happend", m[k], k,  m)
        
        #print("Solution: ", ii)
        #for i in range(0, II):
        #    for n in schedule[i]:
        #        print("\tNode ", n, " Mapped on PE ", node_pe[n], " at time ", i)
        #print("")
        #new_row = ["map" + str(ii + 100000)]
        #for t in range(0, II):  
        #    #print(t)     
        #    for p in range(size_x * size_y):
        #        #print("PE" + str(p) + " at time " + str(t), new_row[-1])
        #        #print(pe_nodes)
        #        
        #        if p in pe_nodes:
        #            found = False
        #            for n in pe_nodes[p]:
        #                if n in schedule[t]:
        #                    new_row.append(int(n))
        #                    found = True
        #                    break
        #            if not found:
        #                new_row.append(-1)
        #        else:
        #            new_row.append(-1)
        #        #print("PE" + str(p) + " at time " + str(t), new_row[-1])
        #new_row.append(0)
        #row.append(new_row)

        #print(new_row)
        
        if ii == 0:
            break
        ii += 1

    #print(ii)
    #with open("csv_invalid_space_negated_mappings", mode='w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(row)

    end = time.time()
    total_time += (end-start)
    print("Time for monomorphism search: " + str(end - start))

    if len(pe_nodes) == 0:
        print("Monomorphism not found!")
        exit(0)

    if check_solution(pe_nodes, dfg, size_y, size_x) == False:
        print("Solution is not correct")
        print()
        exit(0)


    print("Monomorphism found!")

    print("Final mapping")
    for i in range(0, II):
        for n in schedule[i]:
            print("Node ", n, " Mapped on PE ", node_pe[n], " at time ", i)

    print("Total time:", total_time)

def getTime(node, schedule):
    for i in schedule:
        if node in schedule[i]:
            return i

    print("Error while getting scheduling time of node")
    exit(0)

def isConnected2(pe1, pe2, size_y, size_x):

        #return random.randint(0,1)
        i1 = pe1 // size_y
        j1 = pe1 % size_y

        i2 = pe2 // size_y
        j2 = pe2 % size_y

        #same row
        if i1 == i2:
            if (pe1 == pe2 + 1) or (pe1 == pe2 - 1):
                return False
            if abs(pe1 - pe2) == size_y - 1:
                return False

        #same col
        if j1 == j2:
            if (pe1 == pe2 + size_y) or (pe1 == pe2 - size_y):
                return False
            if abs(i1 - i2) == size_x - 1:
                return False
        #center
        if pe1 == pe2:
            return False

        return True

#original below
def isConnected(pe1, pe2, size_y, size_x):

        
        i1 = pe1 // size_y
        j1 = pe1 % size_y

        i2 = pe2 // size_y
        j2 = pe2 % size_y

        #same row
        if i1 == i2:
            if (pe1 == pe2 + 1) or (pe1 == pe2 - 1):
                return True
            if abs(pe1 - pe2) == size_y - 1:
                return True

        #same col
        if j1 == j2:
            if (pe1 == pe2 + size_y) or (pe1 == pe2 - size_y):
                return True
            if abs(i1 - i2) == size_x - 1:
                return True
        #center
        if pe1 == pe2:
            return True

        return False


def get_back_edges(graph):    
    back_edges = []
    edge_attributes = nx.get_edge_attributes(graph, "type")
    for e in edge_attributes:
        if edge_attributes[e] == "back_dep":
            back_edges.append(e)
    return back_edges

def asap_schedule(graph, mode = 1):

    # Remove loop carried dependencies before computing the schedule
    back_edges = get_back_edges(graph)
    for e in back_edges:
        graph.remove_edge(e[0], e[1])


    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph must be a Directed Acyclic Graph (DAG)")
    
    # Topological sorting of the graph
    topo_order = list(nx.topological_sort(graph))
    
    # Initialize the schedule dictionary with start times
    asap_times = {node: 0 for node in topo_order}
    
    # Calculate the ASAP times
    for node in topo_order:
        for pred in graph.predecessors(node):
            asap_times[node] = max(asap_times[node], asap_times[pred] + 1)
    
    # Organize nodes by their ASAP times
    schedule = defaultdict(list)
    for node, time in asap_times.items():
        schedule[time].append(node)
    
    # Restore loop carried dependencies 
    for e in back_edges:
        graph.add_edge(e[0], e[1], type = "back_dep")
    #print("ASAP Schedule len", len(schedule))
    if mode == 1:
        return dict(schedule)
    elif mode == 0:
        return asap_times
  
def alap_schedule(graph, mode = 1):
     
    # Remove loop carried dependencies before computing the schedule
    back_edges = get_back_edges(graph)
    for e in back_edges:
        graph.remove_edge(e[0], e[1])

    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph must be a Directed Acyclic Graph (DAG)")
    
    # Topological sorting of the graph in reverse order
    reverse_topo_order = list(nx.topological_sort(graph))[::-1]
    
    # Find the maximum depth of the graph
    max_depth = len(asap_schedule(graph)) - 1
    #max_depth = len(nx.dag_longest_path(graph))

    # Initialize the schedule dictionary with latest start times
    alap_times = {node: max_depth for node in reverse_topo_order}
    
    # Calculate the ALAP times
    for node in reverse_topo_order:
        for succ in graph.successors(node):
            alap_times[node] = min(alap_times[node], alap_times[succ] - 1)
    
    # Organize nodes by their ALAP times
    schedule = defaultdict(list)
    for node, time in alap_times.items():
        schedule[time].append(node)
    

    # Restore loop carried dependencies 
    for e in back_edges:
        graph.add_edge(e[0], e[1], type = "back_dep")


    if mode == 1:
        return dict(schedule)
    elif mode == 0:
        return alap_times

def mobility_schedule(graph):

    MS = {}
    asap_times = asap_schedule(graph, mode = 0)
    alap_times = alap_schedule(graph, mode = 0)

    for n in graph.nodes:
        t_asap = asap_times[n]
        t_alap = alap_times[n]

        for t in range(t_asap, t_alap + 1):
            if t not in MS:
                MS[t] = []
            MS[t].append(n)

    return MS

def kernel_mobility_schedule(graph, II):
    KMS = {}
    MS = mobility_schedule(graph)
    scheduleLen = len(MS) - 1


    if II <= scheduleLen + 1:
        for i in range(0, scheduleLen + 1):
            it = i//II
            
            if (i%II) not in KMS:
                KMS[i%II] = []
            for nid in MS[i]:
                KMS[i%II].append((it, nid))
    else:
        
        dup = II - (scheduleLen + 1)
        it = 0
        tmpKMS = {}
        for d in range(0, dup + 1):
            for i in range(0, scheduleLen + 1):
                if (i + d) not in tmpKMS:
                    tmpKMS[i + d] = []
                for nid in MS[i]:
                    if nid not in tmpKMS[i + d]:
                        tmpKMS[i + d].append(nid)

            d += 1
            
        for t in tmpKMS:
            if t not in KMS:
                KMS[t] = []
            for nid in tmpKMS[t]:
                KMS[t].append((it, nid))

    #for t in KMS:
    #    print(t, KMS[t])
    return KMS

def generate_valid_schdule(graph, II, array_size, topology_degree):

    #s = Solver()
    back_edges = get_back_edges(graph)
    schedule_result = {}
    valid_schedule = False
    #graph = graph.to_undirected()
    #ss = list(nx.weakly_connected_components(graph))
    #print("weak", len(ss))
    #for sg in ss:
    #    print(len(sg))
    #exit(0)
    while(not valid_schedule):
        s = Solver()
        #print("Scheduling solver timeout set")
        #s.set("timeout", 5*1000) # seconds * 1000
        print("Scheduling II =", II)
        start = time.time()
        KMS = kernel_mobility_schedule(graph, II)
        #for i in range(II):
        #   print(i, KMS[i])
        #for n in graph:
        #    print("Node", n, "has mobility", getMobilityValue(KMS,n))
        #    for succ in list(graph.successors(n)):
        #        print("\tSuccessor", succ, "has mobility", getMobilityValue(KMS, succ))

        fully_encoded = True
        #print(KMS)
        # Generate variables
        iterations = {}
        nit = math.ceil((len(mobility_schedule(graph))) / II)
        for it in range(0, nit):
            if it not in iterations:
                iterations[it] = {}
            for t in KMS:
                if t not in iterations[it]:
                    iterations[it][t] = []
                for p in KMS[t]:
                    if p[0] == it:
                        iterations[it][t].append(p[1])


        literals = []
        for it in iterations:
            for c in iterations[it]:
                for NodeId in iterations[it][c]:                
                    literals.append((Bool("v_%s_%s_%s" % (str(NodeId),str(c),str(it))), NodeId, c, it))

        c_n_it_literal = {}
        n_c_it_literal = {}
        cycle_literals = {}
        node_literals = {}
        for l in literals:
            literal = l[0]
            nodeid = l[1]
            cycle = l[2]
            iteration = l[3]

            if nodeid not in node_literals:
                node_literals[nodeid] = []
            node_literals[nodeid].append(literal)

            if cycle not in cycle_literals:
                cycle_literals[cycle] = []
            cycle_literals[cycle].append(literal)

            if nodeid not in n_c_it_literal:
                n_c_it_literal[nodeid] = {}
            
            if cycle not in n_c_it_literal[nodeid]:
                n_c_it_literal[nodeid][cycle] = {}

            if iteration not in n_c_it_literal[nodeid][cycle]:
                n_c_it_literal[nodeid][cycle][iteration] = literal

            if cycle not in c_n_it_literal:
                c_n_it_literal[cycle] = {}
            

            if nodeid not in c_n_it_literal[cycle]:
                c_n_it_literal[cycle][nodeid] = {}
            
            if iteration not in c_n_it_literal[cycle][nodeid]:
                c_n_it_literal[cycle][nodeid][iteration] = {}
                
            c_n_it_literal[cycle][nodeid][iteration] = literal
        #for t in KMS:
        #    print(t, KMS[t])
        # Start encoding scheduling constraints

        
        for e in graph.edges:
            if e in back_edges:
                continue

            source = e[0]
            destination = e[1]

            #print(source, destination)
            tmp = []
            for (cs, cd) in itertools.product(c_n_it_literal, c_n_it_literal):
                if (source not in c_n_it_literal[cs]) or (destination not in c_n_it_literal[cd]):
                    continue
                
                for (it1, it2) in itertools.product(c_n_it_literal[cs][source], c_n_it_literal[cd][destination]):
                    if it1 == it2 and cd > cs:
                        tmp.append(And(c_n_it_literal[cs][source][it1], c_n_it_literal[cd][destination][it2]))
                    elif abs(it1 - it2) == 1 and it1 < it2 and cd <= cs:
                        tmp.append(And(c_n_it_literal[cs][source][it1], c_n_it_literal[cd][destination][it2]))
                    
            if len(tmp) == 0: 
                print("Not all constraints encoded. II is too small")
                fully_encoded = False
                break
            s.add(Or(tmp))
        #back dep
        for e in graph.edges:
            if e not in back_edges:
                continue
            
            source = e[0]
            destination = e[1]

            #print(source, destination)
            tmp = []
            for (cs, cd) in itertools.product(c_n_it_literal, c_n_it_literal):
                if (source not in c_n_it_literal[cs]) or (destination not in c_n_it_literal[cd]):
                    continue
                
                for (it1, it2) in itertools.product(c_n_it_literal[cs][source], c_n_it_literal[cd][destination]):
                    if abs(it1 - it2) > 1:
                        continue
                    if it1 == it2 and cs > cd:
                        tmp.append(And(c_n_it_literal[cs][source][it1], c_n_it_literal[cd][destination][it2]))
                    elif it1 > it2 and cs < cd:
                        tmp.append(And(c_n_it_literal[cs][source][it1], c_n_it_literal[cd][destination][it2]))
                    elif it1 == it2 and cs == cd:
                        tmp.append(And(c_n_it_literal[cs][source][it1], c_n_it_literal[cd][destination][it2]))
                    
            if len(tmp) == 0: 
                print("Not all constraints encoded")
                fully_encoded = False
                break
            s.add(Or(tmp))

        if not fully_encoded:
            II += 1
            continue


        #exit(0)


        '''
        for n in graph.nodes:
            print("Encoding ",n)
            successors = list(graph.successors(n))
            #print(successors)
            scheduling_combinations = []
            for c in n_c_it_literal[n]:
                for it in n_c_it_literal[n][c]:

                    cycles = [n_c_it_literal[s] for s in successors]
                    cartesian_cycles = itertools.product(*cycles)

                    for cycles_tuple in cartesian_cycles:

                        iterations = []
                        for (cycle, partial_cycles_dict) in zip(cycles_tuple, cycles):
                            iterations.append(partial_cycles_dict[cycle])
                        
                        cartesian_iterations = itertools.product(*iterations)

                        for iterations_tuple in cartesian_iterations:
                            #print(iterations_tuple)
                            if check_placement_feasibility(n, it, c, iterations_tuple, cycles_tuple, successors, back_edges, topology_degree):
                                tmp_comb = [n_c_it_literal[n][c][it]]
                                for (n_d, c_d, it_d) in zip(successors, cycles_tuple, iterations_tuple):
                                    tmp_comb.append(n_c_it_literal[n_d][c_d][it_d])
                                
                                scheduling_combinations.append(And(tmp_comb))
                                #continue
                                                       
            
            if len(scheduling_combinations) > 1:
                s.add(Or(scheduling_combinations))
            elif len(scheduling_combinations) == 1:
                s.add(scheduling_combinations)
            else:
                print("Not all constraints encoded")
                fully_encoded = False

        if not fully_encoded:
            II += 1
            continue
        '''
        # Only one literal must be set to true
        for nodeid in node_literals:
            if len(node_literals[nodeid]) == 1:
                continue
            phi = Or(node_literals[nodeid])
            tmp = []
            for i in range(len(node_literals[nodeid])-1):
                for j in range(i+1, len(node_literals[nodeid])):
                    tmp.append(Not(And(node_literals[nodeid][i], node_literals[nodeid][j])))
            tmp = And(tmp)
            exactlyone = And(phi,tmp)
            s.add(exactlyone)


        
        # Capacity constraints
        # Constraints on number of nodes scheduled at each time step
        # Every time steps cannot contain more than CGRA_X * CGRA_Y nodes
        print(array_size)
        for t in cycle_literals:
            s.add(Sum(cycle_literals[t]) <= array_size)
            s.add(Sum(cycle_literals[t]) >= 1)

        # Connetivity constraints
        for n in graph.nodes:
            successors = list(graph.successors(n))
            if len(successors) > 2:
                for ck in KMS:
                    tmp = []
                    for (it, nid) in KMS[ck]:
                        if nid in successors:
                            tmp.append(c_n_it_literal[ck][nid][it])
                    s.add(Sum(tmp) <= topology_degree)
                #exit(0)



            
            #exit(0)

        # Set node with mobility == 1 to True
        for n in graph:
            if getMobilityValue(KMS, n) == 1:
                s.add(node_literals[n])
                #print("Has mobility 1", n, node_literals[n])
            #print("Node", n, "has mobility", getMobilityValue(KMS,n))
            #for succ in list(graph.successors(n)):
            #    print("\tSuccessor", succ, "has mobility", getMobilityValue(KMS, succ))
            

        
        
        end = time.time()
        print("Time to generate constraints: " + str(end - start))

        print("Start solving...")
        start = time.time()
        
        #with open("Schedule_II_"+str(II), "w") as f:
        #    f.write(s.to_smt2())


        if s.check() == sat:
            #print("SAT")
            m = s.model()
            #
            #valid_schedule = True
            ##print(m)
            #for t in m.decls():
            #    #print(t, m[t])
            #    if is_true(m[t]):
            #        tmp = str(t).split('_')
            #        nodeid = int(tmp[1])
            #        cycle = int(tmp[2])
            #        iteration = int(tmp[3])
            #
            #        if cycle not in schedule_result:
            #            schedule_result[cycle] = []
            #        schedule_result[cycle].append(nodeid)
            model_number = 0
            while s.check() == sat:
                
                print("MODEL " + str(model_number))
                model_number+=1
                m = s.model()
                block = []  
                for z3_decl in m: # FuncDeclRef
                    arg_domains = []
                    for i in range(z3_decl.arity()):
                        domain, arg_domain = z3_decl.domain(i), []
                        for j in range(domain.num_constructors()):
                            arg_domain.append( domain.constructor(j) () )
                        arg_domains.append(arg_domain)
                    for args in itertools.product(*arg_domains):
                        block.append(z3_decl(*args) != m.eval(z3_decl(*args)))
                s.add(Or(block))
                if model_number == 4:#4
                    break
            #m = s.model()
            # 5x5 II 3
            #[ 16 3 25 40 47 4 10 29 34 0 20 46 43 28 7 50 11 38 ]
            #[ 30 21 39 49 37 44 26 41 5 23 17 19 32 1 8 14 35 12 ]
            #[ 42 13 33 24 22 48 6 27 15 36 9 18 2 45 31 ]       
            valid_schedule = True
            #print(m)
            for t in m.decls():
                #print(t, m[t])
                if is_true(m[t]):
                    tmp = str(t).split('_')
                    nodeid = int(tmp[1])
                    cycle = int(tmp[2])
                    iteration = int(tmp[3])
                        #
                    if cycle not in schedule_result:
                        schedule_result[cycle] = []
                    schedule_result[cycle].append(nodeid)
        else:
            print("UNSAT")   
        end = time.time()
        print("Time to find schedule: " + str(end - start))
        II += 1

    

    # Schedule feasibility constraints
    #for c in scheduling_combinations:

    return schedule_result

def check_placement_feasibility(source, it_source, cycle_source, iterations_tuple, cycles_tuple, successors, back_edges, topology_degree):
    

    # Discard tuple if the distance between the source node iteration and
    # the destination node iteration is greater then 1
    for it_t in iterations_tuple:
        if abs(it_source - it_t) > 1:
            return False

    # Evaluate topology degree
    top_deg = {i: cycles_tuple.count(i) for i in cycles_tuple}
    #print(cycles_tuple)
    for k in top_deg:
        if top_deg[k] > topology_degree:
            return False        

    # Modulo scheduling feasibility constraints - check SAT paper for more info
    for (it_t, c_t, destination) in zip(iterations_tuple, cycles_tuple, successors):

        if (source, destination) in back_edges:
            if c_t <= cycle_source and it_t == it_source:
                continue
            elif c_t > cycle_source and it_t != it_source:
                continue
            else:
                return False
        else:
            if c_t > cycle_source and it_t == it_source:
                continue
            elif c_t <= cycle_source and it_t != it_source:
                continue
            else:
                return False

    return True


def main():

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-path', type=str, help='Input file containing the DFG')
    parser.add_argument('-x', type=int, help='Number or rows in the CGRA (default value: 4)', default=4)
    parser.add_argument('-y', type=int, help='Number or rows in the CGRA (default value: 4)', default=4)
    parser.add_argument('-d', type=int, help='Topology degree (default value: 5)', default=5)
    parser.add_argument('-i', type=int, help='Iteration Itnerval (default value: -1)', default=-1)
    

    #Parse Arguments
    args = parser.parse_args()

    edgefile = args.path

    #thread_timeout = int(args.t)

    CGRA_X = int(args.x)
    CGRA_Y = int(args.y)

    CGRA_SIZE = CGRA_X * CGRA_Y
    print(CGRA_SIZE, "CGRA size")
    arch = nx.Graph()
    dfg  = nx.DiGraph()

    topology_degree = int(args.d)


    print(edgefile)
    print(CGRA_X, "x", CGRA_Y)


    # Parse input DFG
    with open(edgefile,"r") as fd: 
        for l in fd.read().splitlines():
            edge = [int(x) for x in l.split(' ')]
            if edge[2] == 1:
                dfg.add_edge(edge[0], edge[1], type="back_dep")
            else:
                dfg.add_edge(edge[0], edge[1], type="data_dep")
    print("Parsing done!")
    #dfg = G
    
    # Get starting II
    print("NOTE: The architecture time is not included in the final compilation time.")
    RecII = 1
    ResII = math.ceil(len(dfg.nodes) / CGRA_SIZE)
    for ec in nx.recursive_simple_cycles(dfg):
        RecII = max(RecII, len(ec))
        print("EC",ec)
    print("RecII is computed with recursive_simple_cycles from networkX.\nSometimes it doesn't provide the correct lowerbound.\nTo manually set the II , use the -i option.")
    # Should be max(recII, resII), but there is a bug in some cases.
    # SAT-MapIt computes the correct lowerbound
    # Only looking at the length of the recursive simple cycle
    # of the DFG is not correct 

    II = max(RecII, ResII)
    print("II = max(RecII, ResII) = max(" + str(RecII) + ", " + str(ResII) + ")= " + str(II))
    print("#nodes: ",len(dfg.nodes))
    print("#edges: ",len(dfg.edges))
    print("#maxdegree: ", getMaxOutDegree(dfg))
    if int(args.i) != -1:
        II = int(args.i)
        print("Manually setting II to", II)


    map(dfg, arch, II, topology_degree, CGRA_Y, CGRA_X)

    



if __name__ == "__main__":
    main()



