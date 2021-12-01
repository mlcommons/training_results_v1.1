
import os


class SimpleWrapper:
    def __init__(self, args, mllog):

        # Only log on one instance (if in distributed mode)
        self.should_log = (not args.use_popdist or args.popdist_rank == 0)
        if self.should_log:
            self.mllogger = mllog.get_mllogger()
            filename = f"results/{args.submission_platform}/result_{args.submission_run_index}.txt"
            if not os.path.exists("results"):
                os.mkdir("results")
            if not os.path.exists(f"results/{args.submission_platform}"):
                os.mkdir(f"results/{args.submission_platform}")
            if os.path.exists(filename):
                os.remove(filename)
            mllog.config(filename=filename)


    def start(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.start(*args, **kwargs)


    def end(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.end(*args, **kwargs)


    def event(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.event(*args, **kwargs)