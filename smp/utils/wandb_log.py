import wandb

cls_list = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

class Wandb:
    def __init__(self, project, flag, name, config):
        self.flag = flag
        if self.flag:
            wandb.init(project=project, name=name, entity="cv-3-bitcoin", config=config)
    
    def write_train(self, mIoU, acc, loss):
        if self.flag:
             wandb.log({
                'train_mIoU':mIoU,
                'train_acc':acc,
                'train_loss':loss.item()
            })

    def write_val(self, avrg_loss, val_mIoU):
        if self.flag:
            wandb.log({
                    'val_loss' : avrg_loss,
                    'val_mIoU' : val_mIoU,
                })

    def write_class(self, IoU_by_class):
        if self.flag:
            log_dict = dict([ [cls_list[i], IoU_by_class[i][cls_list[i]] ] for i in range(11)])
            wandb.log(log_dict)

    def write_lr(self, epoch, current_lr, val_loss, val_acc, f1):
        if self.flag:
            wandb.log({"epoch": epoch, "learning rate": current_lr, "Val/loss": val_loss,"Val/accuracy": val_acc, "F1 Score": f1})


    def watch(self, model):
        if self.flag:
            wandb.watch(model)