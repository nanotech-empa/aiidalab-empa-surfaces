import numpy as np
def compute_cost(element_list):
    cost={'H':1,'C':4,'Si':4,'N':5,'O':6,'Au':11,'Cu':11,'Ag':11,'Pt':18,'Co':11,'Zn':10}
    the_cost=0
    for element in element_list:
        s = ''.join(i for i in element if not i.isdigit())
        if isinstance(s[-1] ,type(1)):
            s=s[:-1]
        if s in cost.keys():
            the_cost+=cost[s]
        else:
            the_cost+=4
    return the_cost


def compute_nodes(cost,tasks_per_node):
    tasks=max(tasks_per_node,int(324*cost/10000))
    list_squares=[i*tasks_per_node  for i in range(1,200) 
                  if int(np.sqrt(i*tasks_per_node)) * int(np.sqrt(i*tasks_per_node)) == i*tasks_per_node ]
    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
    ideal = int(takeClosest(tasks,list_squares) / tasks_per_node )
    ratio = ideal*tasks_per_node / tasks
    result = ideal
    if ratio > 1.3 or ratio < 0.7:
        result = int(tasks/tasks_per_node)
    return result 