import torch
import config_pix2pix
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config_pix2pix.DEVICE), y.to(config_pix2pix.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake *0.5 *0.5  #remove Normalisation
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder +f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y *0.5, + 0.5, folder +f"/label_{epoch}.png")
    gen.train()



def save_checkpoint(model, optimizer, filename ="my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    checkpoint ={
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)





































