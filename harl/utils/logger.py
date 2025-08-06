import wandb
from tensorboardX import SummaryWriter
import numpy as np
import datetime

class Logger:
    def __init__(self, args, _log_dir =  None) -> None:
        time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args = args
        if self.args["tb_plot"]:
            self.writer = SummaryWriter(log_dir=_log_dir)           
            
            
        if self.args["wandb"]:
            wandb.init(project="acorm", 
                       group = '{}_{}{}'.format(self.args.env_name, self.args.algorithm, self.args.tag),
                       name = 'seed{}_date{}'.format(self.args.seed, time_path),
                       )
            # Add extra arguments to existing args under separate header
            wandb.config.update(vars(args))
            
            if _log_dir is not None:
                wandb.config.update(vars(_log_dir)['_content'], allow_val_change=True)
            
        
        self.step = 0
        self.episode = 0
    
    
    def log(self, log_ls, step):
        for idx, log in enumerate(log_ls):
            if self.args["tb_plot"]:

                if isinstance(log, dict):
                    for key, value in log.items():
                        self.writer.add_scalar(key + f"_{idx}", value, step)
                else:
                    raise ValueError("log should be a dictionary")

            if self.args["wandb"]:
                wandb.log(log, step)
    
    def log_video(self, video, step):
        if self.args["wandb"]:
            video = [np.transpose(np.array(f), (2, 0, 1)) for f in video]
            video = np.stack(video, axis = 0)
            print
            wandb.log({"video": wandb.Video(video)}, step=step)
            
    def cleanup(self):
        if self.args["tb_plot"]:
            self.writer.close()
        if self.args["wandb"]:
            wandb.finish()