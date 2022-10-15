from stringprep import in_table_c3
import numpy as np
import torch.nn as nn
import torch
import utils
import resnet18 as networks
import dataset_builder as db
###############TODO
#add a logger where to save the parameters and train results
###############TODO

# IMPORT HYPER PARAMETERS AND PARSE 
TRAIN = True
TEST_ON_TRAIN = False
TESTING_FREQ = 5

########
### split test and train
########

batch_size = 124
validation_split_index = .2
seed= 42
np.random.seed(seed)

# Dataset
C = db.Cards("./annotation.json", "./dataset/Images/Images")

# Creating data indices for training and validation split_indexs:
dataset_size = C.__len__()
print("dataset size is : ", C.__len__())
indices = list(range(dataset_size))
split_index = int(np.floor(validation_split_index * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split_index:], indices[:split_index]

# Creating data samplers and loaders:
train_sampler = db.SubsetRandomSampler(train_indices)
valid_sampler = db.SubsetRandomSampler(val_indices)

train_loader = db.DataLoader(C, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=False)
validation_loader = db.DataLoader(C, batch_size=batch_size, sampler=valid_sampler, num_workers=1, pin_memory=False)


# TEST and TRAIN functions

def test(model, data_loader, device ):
    '''this tests one given model, passed as a model and not a path'''
    acc = 0
    for data, label in data_loader:
        data = data.type(torch.FloatTensor) 
        data = data.to(device)
        model = model.eval()
        out = model(data)
        acc = 0
        for i in range(len(data)):
            if torch.argmax(out[i]) == torch.argmax(label[i]):
                acc += 1
        
    acc = acc/len(data_loader.dataset)
    return acc 

def pretrain(model):
    #TODO, might be missing arguments
    pass 

def train_model(model, num_epochs, train_loader, test_loader, device, save_location = "./log.txt"): 
    #TODO , make a better logger
    '''this function trains a model'''
    save_name = utils.log_hyper_param("./log")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    #schedueler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 15, gamma = 0.1)
    model.train()
    f = open(save_location, 'a') 
    acc_history = []
    for epoch in range(num_epochs):
        j=0
        acc = 0
        #training on one epoch
        for data, label in train_loader:
            data = data.type(torch.FloatTensor) 
            print("data shape :::::::::::::::: ", data.size())
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            criterion = torch.nn.MultiLabelSoftMarginLoss()
            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad() #resets the grad attribute of the different parameters
            #schedueler.step()  #decays lr every n epochs

            #calc accuracy 
            for i in range(len(data)):
                if torch.argmax(out[i]) == torch.argmax(label[i]):
                    acc += 1
            acc += acc / len(data)
            print("len data = ", len(data) ," training accuracy on batch", j,  " : ", acc)
            j+= 1
        acc_history.append(acc)
        f.writelines(str(acc_history)+'\n')
        

        #testing at the end of the epoch
        if epoch%TESTING_FREQ == 0 :
            test_acc = test(model, test_loader, device=device)
            print("testing accuracy : ", test_acc)
            f.writelines(str(acc_history)+'\n')
        
        def save_network(network, save_path= "model_epoch_"+str(epoch)):
            torch.save(network.state_dict(), save_path)
        
        save_network(model)

        


        


def main():   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = networks.resnet18(pretrained=True)


    if TRAIN == True:
        train_model(model, num_epochs=50, train_loader=train_loader, test_loader=validation_loader, device=device)

if __name__ == "__main__" :
    main()
