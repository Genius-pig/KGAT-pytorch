import os
from collections import OrderedDict

import numpy as np
import torch


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def save_embeddings(user_embed, item_embed, embed_path, current_epoch, last_best_epoch=None):
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)
    embed_state_file = os.path.join(embed_path, 'mf{}'.format(current_epoch))
    user_tensor = user_embed.weight.cpu().data
    item_tensor = item_embed.weight.cpu().data
    np.savez(embed_state_file, user_embed=user_tensor, item_embed=item_tensor)
    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(embed_path, 'mf{}.npz'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))
