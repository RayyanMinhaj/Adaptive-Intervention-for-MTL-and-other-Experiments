import torch
import torch.nn as nn
import torch.nn.functional as F

class AIMPolicy(nn.Module):
    def __init__(self, num_tasks, policy_type='matrix', temp_k=10.0):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.policy_type = policy_type
        self.k = temp_k


        # initializing learnabel threshold (tau)
        # scalar tau is just a single number but a matrix is NxN
        if policy_type == 'matrix':
            self.tau = nn.Parameter(torch.zeros((num_tasks, num_tasks)))

        elif policy_type == 'scalar':
            self.tau = nn.Parameter(torch.zeros(1))

    

    def get_intervened_gradient(self, raw_task_gradients):
        # input is the list of gradient tensors for each task [g1, g2, g3....]
        # output is the final summed gradient vector g_intervened

        # Stack gradients for easier math: Shape (Num_Tasks, Param_Dim)
        # We assume flattened gradients for simplicity here
        grads_stack = torch.stack(raw_task_gradients) 
        num_tasks = len(raw_task_gradients)

        g_prime = grads_stack.clone()

        # INTERVENTION LOOP (8-18 in A.1)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue # dont really need to compare tasks with themselves

                g_i =  grads_stack[i]
                g_j =  grads_stack[j]

                # 1. Compute Cosine Similarity
                norm_i = torch.norm(g_i) + 1e-8
                norm_j = torch.norm(g_j) + 1e-8
                cos_sim = torch.dot(g_i, g_j) / (norm_i * norm_j)

                # 2. Threshhold Tau
                if self.policy_type == 'matrix':
                    tau_ij = self.tau[i, j]
                
                elif self.policy_type == 'scalar':
                    tau_ij = self.tau[0]


                # 3. Compute Projection Weight (eq. 1)
                # w = sigmoid(k * (tau - cos_sim))

                w_proj = torch.sigmoid(self.k * (tau_ij - cos_sim))


                # 4. Compute Vector Projecction of g_i onto g_j
                # proj = (g_i . g_j)/ ||g_j||^2 * g_j

                proj_factor = torch.dot(g_i, g_j) / (norm_j**2)
                projection = proj_factor * g_j


                # 5. Modify gradient (eq. 2)
                # we subtract the weighted projection
                # but remember: we modify g_prime[i] (the accumulator) using raw g_j
                g_prime[i] = g_prime[i] - w_proj * projection

        
        # finally, sum all modified gradients to get the final update direction
        g_intervened = torch.sum(g_prime, dim=0)

        return g_intervened




