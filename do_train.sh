# Default setting:
cmd='python -m cmr.experiments.shape --name=bird_net --display_port 8087'

# Stronger texture & higher resolution texture.
cmd='python -m cmr.experiments.shape --name=bird_net_better_texture --tex_size=6 --tex_loss_wt 1. --tex_dt_loss_wt 1. --display_port 8088'

# Stronger texture & higher resolution texture + higher res mesh. 
cmd='python -m cmr.experiments.shape --name=bird_net_hd --tex_size=6 --tex_loss_wt 1. --tex_dt_loss_wt 1. --subdivide 4 --display_port 8089'
