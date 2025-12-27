import torch
import torch.nn as nn

class SimpleMTLModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_tasks):
        super().__init__()

        self.num_tasks = num_tasks

        #shared backbone - all tasks share same weights 
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # task specific heads - each task gets its own specific head to predict its specific value
        #self.heads = nn.ModuleList([
        #    nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        #])

        self.heads = nn.ModuleList()

        for task in range(num_tasks):
            head = nn.Linear(hidden_dim, 1)

            self.heads.append(head)




    def forward(self, x):
        shared_rep = self.backbone(x)

        outputs = [] # individual task outputs
        

        for task in range(self.num_tasks):
            task_output = self.heads[task](shared_rep)
            outputs.append(task_output)

        return outputs