import torch
import torch.optim.lr_scheduler as lr_scheduler

class YOLOV1LrScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        """
        Implementation of the scheduler described in the original paper of YOLO v1.
        It increases the lr from 0.001 up to 0.01 in the first 30 epochs.
        It keeps the lr as 0.01 for the next 75 epochs.
        It keeps the lr as 0.001 for the next 30 epochs.
        It keeps the lr as 0.0001 for the rest of the training epochs.

        Inputs: 
            >> optimizer: () Optimizer used during the training of the model.
        """
        super(YOLOV1LrScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        epoch = self.last_epoch
        lr = self.base_lrs[0]
        
        if epoch <= 75:
            m = ((1e-3 - 1e-2) / (75 - 0)) 
            b = (1e-2) - m * 0
            lr = m*epoch + b
            #lr = 1e-2
        elif epoch <= 105:
            m = ((1e-4 - 1e-3) / (105 - 75))
            b = (1e-3) - m * 75
            lr = m * epoch + b
            #lr = 1e-3
        else:
            lr = 1e-4
        
        return [lr]

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    model = torch.nn.Linear(10,100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = YOLOV1LrScheduler(optimizer)
    
    criterion = torch.nn.MSELoss()
    labels = torch.rand(100)

    lr_values = []
    for epoch in range(140):
        for _ in range(10):
            input = torch.rand(10)
            out = model(input)

            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning Rate = {lr}")
        lr_values.append(lr)

    plt.plot(lr_values)
    plt.show()
