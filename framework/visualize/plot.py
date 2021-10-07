from matplotlib.pyplot import hist, plot, text
import torch
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from ..utils import U
from typing import Dict, Tuple, List, Optional, Callable, Union
import threading
import atexit
from torch.multiprocessing import Process, Queue, Event
from queue import Empty as EmptyQueue
from queue import Full as FullQueue
import sys
import itertools
import PIL
import time

wandb = None
plt = None
make_axes_locatable = None
FigureCanvas = None


def import_matplotlib():
    global plt
    global make_axes_locatable
    global FigureCanvas
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.backends.backend_agg import FigureCanvas


class CustomPlot:
    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        pass

    def to_wandb(self):
        return None


class Histogram(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], n_bins: int = 64):
        if torch.is_tensor(data):
            data = data.detach().cpu()

        self.data = data
        self.n_bins = n_bins

    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        summary_writer.add_histogram(name, self.data, global_step, max_bins=self.n_bins)

    def to_wandb(self):
        return wandb.Histogram(self.data, num_bins=self.n_bins)


class Image(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], caption: Optional[str] = None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        self.data = data.astype(np.float32)
        self.caption = caption

    def to_tensorboard(self, name, summary_writer, global_step):
        if self.data.shape[-1] in [1,3]:
            data = np.transpose(self.data, (2,0,1))
        else:
            data = self.data
        summary_writer.add_image(name, data, global_step)

    def to_wandb(self):
        if self.data.shape[0] in [1, 3]:
            data = np.transpose(self.data, (1,2,0))
        else:
            data = self.data

        data = PIL.Image.fromarray(np.ascontiguousarray((data*255.0).astype(np.uint8)), mode="RGB")
        return wandb.Image(data, caption = self.caption)


class Scalars(CustomPlot):
    def __init__(self, scalar_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]]):
        self.values = {k: v.item() if torch.is_tensor(v) else v for k, v in scalar_dict.items()}
        self.leged = sorted(self.values.keys())

    def to_tensorboard(self, name, summary_writer, global_step):
        v = {k: v for k, v in self.values.items() if v == v}
        summary_writer.add_scalars(name, v, global_step)

    def to_wandb(self):
        return self.values


class Scalar(CustomPlot):
    def __init__(self, val: Union[torch.Tensor, np.ndarray, int, float]):
        if torch.is_tensor(val):
            val = val.item()

        self.val = val

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_scalar(name, self.val, global_step)

    def to_wandb(self):
        return self.val


class MatplotlibPlot(CustomPlot):
    def __init__(self):
        import_matplotlib()

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.matplotlib_plot(), global_step)

    def to_wandb(self):
        return self.matplotlib_plot()


class Barplot(MatplotlibPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], names: Optional[List[str]] = None):
        super().__init__()

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        assert data.ndim == 1
        assert (names is None) or (data.shape[0] == len(names))

        self.data = data.tolist()
        self.names = names

    def matplotlib_plot(self):
        f = plt.figure()

        plt.bar([i for i in range(len(self.data))], self.data)
        plt.xticks(self.names)

        return f


class XYChart(MatplotlibPlot):
    def __init__(self, data: Dict[str, List[Tuple[float, float]]], markers: List[Tuple[float,float]] = [],
                 xlim = (None, None), ylim = (None, None)):
        super().__init__()

        self.data = data
        self.xlim = xlim
        self.ylim = ylim
        self.markers = markers

    def matplotlib_plot(self):
        f = plt.figure()
        names = list(sorted(self.data.keys()))

        for n in names:
            plt.plot([p[0] for p in self.data[n]], [p[1] for p in self.data[n]])

        if self.markers:
            plt.plot([a[0] for a in self.markers], [a[1] for a in self.markers], linestyle='', marker='o',
                 markersize=2, zorder=999999)

        plt.legend(names)
        plt.ylim(*self.xlim)
        plt.xlim(*self.ylim)

        return f


class Heatmap(MatplotlibPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: str, ylabel: str,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None, textval: bool = True, subsample_ticks:int = 1,
                 cmap = "auto", colorbar: bool = True):

        super().__init__()
        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks
        self.textval = textval
        self.subsample_ticks = subsample_ticks
        self.cmap = plt.cm.Blues if cmap=="auto" else cmap
        self.colorbar = colorbar

    def get_marks(self, m: Optional[Union[str, List[str]]], n: int):
        if m is None:
            return None

        assert len(m) == n
        return [l for i, l in enumerate(m) if i % self.subsample_ticks == 0]

    def matplotlib_plot(self):
        figure, ax = plt.subplots()
        figure.set_tight_layout(True)

        im = plt.imshow(self.map.astype(np.float32), interpolation='nearest', cmap=self.cmap, aspect='auto')

        x_marks = self.get_marks(self.x_marks, self.map.shape[1])
        y_marks = self.get_marks(self.y_marks, self.map.shape[0])

        if x_marks is not None:
            plt.xticks(np.arange(self.map.shape[1])[::self.subsample_ticks], x_marks, rotation=45, fontsize=8,
                       ha="right", rotation_mode="anchor")

        if y_marks is not None:
            plt.yticks(np.arange(self.map.shape[0])[::self.subsample_ticks], y_marks, fontsize=8)

        if self.textval:
            # Use white text if squares are dark; otherwise black.
            threshold = self.map.max() / 2.

            rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
            for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
                color = "white" if self.map[i, j] > threshold else "black"
                plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        if self.colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=0.1)
            plt.colorbar(im, cax)

        return figure


class AnimatedHeatmap(CustomPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: str, ylabel: str,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None, textval: bool = True, subsample_ticks:int = 1,
                 fps: float = 2, cmap = "auto", colorbar: bool = True, colorbar_ticks = None,
                 colorbar_labels = None, ignore_wrong_marks: bool = False):

        super().__init__()
        import_matplotlib()

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks
        self.textval = textval
        self.subsample_ticks = subsample_ticks
        self.fps = fps
        self.cmap = plt.cm.Blues if cmap=="auto" else cmap
        self.colorbar = colorbar
        self.colorbar_ticks = colorbar_ticks
        self.colorbar_labels = colorbar_labels
        self.ignore_wrong_marks = ignore_wrong_marks

        assert (colorbar_labels is None) == (colorbar_ticks is None)

    def get_marks(self, m: Optional[Union[str, List[str]]], n: int):
        if m is None:
            return None

        if self.ignore_wrong_marks and len(m) != n:
            return None

        assert len(m) == n
        return [l for i, l in enumerate(m) if i % self.subsample_ticks == 0]

    def to_video(self):
        data = self.map.astype(np.float32)

        x_marks = self.get_marks(self.x_marks, self.map.shape[2])
        y_marks = self.get_marks(self.y_marks, self.map.shape[1])

        figure, ax = plt.subplots()
        canvas = FigureCanvas(figure)
        figure.set_tight_layout(True)

        im = plt.imshow(data[0], interpolation='nearest', cmap=self.cmap, aspect='auto', animated=True, 
                        vmin = data.min(), vmax=data.max())

        if x_marks is not None:
            plt.xticks(np.arange(self.map.shape[2])[::self.subsample_ticks], x_marks, rotation=45, fontsize=8,
                    ha="right", rotation_mode="anchor")

        if y_marks is not None:
            plt.yticks(np.arange(self.map.shape[1])[::self.subsample_ticks], y_marks, fontsize=8)

        title = plt.title("Step: 0")

        if self.textval:
            # Use white text if squares are dark; otherwise black.
            threshold = self.map.max() / 2.

            rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
            for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
                color = "white" if self.map[i, j] > threshold else "black"
                plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        if self.colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=0.1)
            cbar = plt.colorbar(im, cax, ticks=self.colorbar_ticks)

            if self.colorbar_labels is not None:
                cbar.ax.set_yticklabels(self.colorbar_labels)


        frames = []
        for i in range(data.shape[0]):
            canvas.draw()
            image_from_plot = np.array(canvas.renderer.buffer_rgba())
            frames.append(image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (4,))[:,:,:3])

            if i < data.shape[0] - 1:
                im.set_data(data[i + 1])
                title.set_text(f"Step: {i + 1}")

        del figure

        video = np.stack(frames, 0)
        return np.transpose(video, (0, 3, 1, 2))

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_video(name, self.to_video()[np.newaxis], global_step, fps = self.fps)

    def to_wandb(self):
        return wandb.Video(self.to_video(), fps = self.fps)


class ConfusionMatrix(Heatmap):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], class_names: Optional[List[str]] = None,
                 x_marks: Optional[List[str]] = None, y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        map = np.transpose(map, (1, 0))
        map = map.astype('float') / map.sum(axis=1).clip(1e-6, None)[:, np.newaxis]

        if class_names is not None:
            assert x_marks is None and y_marks is None
            x_marks = y_marks = class_names

        super().__init__(map, "predicted", "real", round_decimals=2, x_marks = x_marks, y_marks = y_marks)


class TextTable(CustomPlot):
    def __init__(self, header: List[str], data: List[List[str]]):
        self.header = header
        self.data = data

    def to_markdown(self):
        res = " | ".join(self.header)+"\n"
        res += " | ".join("---" for _ in self.header)+"\n"
        return res+"\n".join([" | ".join([x.replace("|", "&#124;") for x in l]) for l in self.data])

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_text(name, self.to_markdown(), global_step)

    def to_wandb(self):
        return wandb.Table(data=self.data, columns=self.header)


class PlotAsync:
    @staticmethod
    def worker(self, fn, *args):
        try:
            self.result = fn(*args)
        except:
            self.failed = True
            raise

    def __init__(self, fn: Callable[[any], Dict[str, any]], args: Tuple=()):
        self.result = None
        self.failed = False

        args = U.apply_to_tensors(args, lambda x: x.detach().cpu().clone())

        self.thread = threading.Thread(target = self.worker, args=(self, fn, *args), daemon=True)
        self.thread.start()

    def get(self, wait: bool) -> Optional[Dict[str, any]]:
        if (self.result is None and not wait) or self.failed:
            return None

        self.thread.join()
        return self.result


class Logger:
    @staticmethod
    def parse_switch_string(s: str) -> Tuple[bool,bool]:
        s = s.lower()
        if s=="all":
            return True, True
        elif s=="none":
            return False, False

        use_tb, use_wandb =  False, False
        s = s.split(",")
        for p in s:
            if p=="tb":
                use_tb = True
            elif p=="wandb":
                use_wandb = True
            else:
                assert False, "Invalid visualization switch: %s" % p

        return use_tb, use_wandb

    def create_loggers(self):
        self.is_sweep = False
        self.wandb_id = {}
        global wandb

        if self.use_wandb:
            import wandb
            wandb.init(**self.wandb_init_args)
            self.wandb_id = {
                "sweep_id": wandb.run.sweep_id,
                "run_id": wandb.run.id
            }
            self.is_sweep = bool(wandb.run.sweep_id)
            wandb.config["is_sweep"] = self.is_sweep
            wandb.config.update(self.wandb_extra_config)

            self.save_dir = os.path.join(wandb.run.dir)

        os.makedirs(self.save_dir, exist_ok=True)
        self.tb_logdir = os.path.join(self.save_dir, "tensorboard")

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tb_logdir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.tb_logdir, flush_secs=30)
        else:
            self.summary_writer = None

    def __init__(self, save_dir: Optional[str] = None, use_tb: bool = False, use_wandb: bool = False,
                 get_global_step: Optional[Callable[[], int]] = None, wandb_init_args={}, wandb_extra_config={}):
        global plt
        global wandb

        import_matplotlib()

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.save_dir = save_dir
        self.get_global_step = get_global_step
        self.wandb_init_args = wandb_init_args
        self.wandb_extra_config = wandb_extra_config

        self.create_loggers()

    def flatten_dict(self, dict_of_elems: Dict) -> Dict:
        res = {}
        for k, v in dict_of_elems.items():
            if isinstance(v, dict):
                v = self.flatten_dict(v)
                for k2, v2 in v.items():
                    res[k+"/"+k2] = v2
            else:
                res[k] = v
        return res

    def get_step(self, step: Optional[int] = None) -> Optional[int]:
        if step is None and self.get_global_step is not None:
            step = self.get_global_step()

        return step

    def log(self, plotlist: Union[List, Dict, PlotAsync], step: Optional[int] = None):
        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p.get(True) if isinstance(p, PlotAsync) else p for p in plotlist if p]
        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        d = {}
        for p in plotlist:
            d.update(p)

        self.log_dict(d, step)

    def log_dict(self, dict_of_elems: Dict, step: Optional[int] = None):
        dict_of_elems = self.flatten_dict(dict_of_elems)

        if not dict_of_elems:
            return

        dict_of_elems = {k: v.item() if torch.is_tensor(v) and v.nelement()==1 else v for k, v in dict_of_elems.items()}
        dict_of_elems = {k: Scalar(v) if isinstance(v, (int, float)) else v for k, v in dict_of_elems.items()}

        step = self.get_step(step)

        if self.use_wandb:
            wandbdict = {}
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v = v.to_wandb()
                    if v is None:
                        continue

                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            wandbdict[k+"/"+k2] = v2
                    else:
                        wandbdict[k] = v
                elif isinstance(v, plt.Figure):
                    wandbdict[k] = v
                else:
                    assert False, f"Invalid data type {type(v)}"

            wandbdict["iteration"] = step
            wandb.log(wandbdict)

        if self.summary_writer is not None:
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v.to_tensorboard(k, self.summary_writer, step)
                elif isinstance(v, plt.Figure):
                    self.summary_writer.add_figure(k, v, step)
                else:
                    assert False, f"Unsupported type {type(v)} for entry {k}"

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def flush(self):
        pass

    def finish(self):
        pass


class AsyncLogger(Logger):
    def redirect(self, f, tag):
        old_write = f.write
        old_writelines = f.writelines
        def new_write(text):
            old_write(text)
            self.print_queue.put((tag, text))
        def new_writelines(lines):
            old_writelines(lines)
            self.print_queue.put((tag, os.linesep.join(lines)))
        f.write = new_write
        f.writelines = new_writelines
        return f

    def wandb_flush_io(self):
        if not self.use_wandb:
            pass

        while not self.print_queue.empty():
            tag, text = self.print_queue.get()
            wandb.run._redirect_cb("stdout", text)
            # wandb.run._redirect_cb("stderr" if tag==1 else "stdout", text)

    @staticmethod
    def log_fn(self, stop_event: Event):
        try:
            self._super_create_loggers()
            self.resposne_queue.put({k: self.__dict__[k] for k in ["save_dir", "tb_logdir", "is_sweep", "wandb_id"]})

            while True:
                self.wandb_flush_io()

                try:
                    cmd = self.draw_queue.get(True, 0.1)
                except EmptyQueue:
                    if stop_event.is_set():
                        break
                    else:
                        continue

                self._super_log(*cmd)
                self.resposne_queue.put(True)
        except:
            print("Logger process crashed.")
            raise
        finally:
            try:
                self.wandb_flush_io()
            except:
                pass

            print("Logger: syncing")
            if self.use_wandb:
                wandb.join()

            stop_event.set()
            print("Logger process terminating...")

    def create_loggers(self):
        self._super_create_loggers = super().create_loggers
        self.stop_event = Event()
        self.stop_requested = False
        self.proc = Process(target=self.log_fn, args=(self, self.stop_event))
        self.proc.start()

        atexit.register(self.finish)

    def __init__(self, *args, queue_size: int = 1000, **kwargs):
        self.queue = []

        self.queue_size = queue_size
        self.print_queue = Queue()
        self.draw_queue = Queue(queue_size)
        self.resposne_queue = Queue()
        self._super_log = super().log
        self.waiting = 0

        super().__init__(*args, **kwargs)

        self.__dict__.update(self.resposne_queue.get(True))
        
        if self.use_wandb:
            # monkey-patch stdout and stderr such that we can redirect it to wandb running in the other process
            sys.stdout = self.redirect(sys.stdout, 0)
            sys.stderr = self.redirect(sys.stderr, 1)

    def log(self, plotlist, step=None):
        if self.stop_event.is_set() or not self.proc.is_alive():
            assert self.stop_requested, "Logger process crashed, but trying to log"
            return

        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        plotlist = U.apply_to_tensors(plotlist, lambda x: x.detach().cpu())

        if step is None:
            step = self.get_global_step()

        self.queue.append((plotlist, step))
        self.flush(wait = False)

    def enqueue(self, data, step: Optional[int]):
        while True:
            if not self.proc.is_alive():
                return

            try:
                self.draw_queue.put((data, step), timeout=1)
                break
            except TimeoutError:
                pass
            except FullQueue:
                time.sleep(0.1)
                pass

        self.waiting += 1

    def wait_logger(self, wait = False):
        cond = (lambda: not self.resposne_queue.empty()) if not wait else (lambda: self.waiting>0)
        already_printed = False
        while cond() and not self.stop_event.is_set() and self.proc.is_alive():
            will_wait = self.resposne_queue.empty()
            if will_wait and not already_printed:
                already_printed = True
                sys.stdout.write("Warning: waiting for logger... ")
                sys.stdout.flush()
            try:
                self.resposne_queue.get(True, 0.2)
            except EmptyQueue:
                continue
            self.waiting -= 1

        if already_printed:
            print("done.")

    def request_stop(self):
        self.stop_requested = True
        self.stop_event.set()

    def flush(self, wait: bool = True):
        while self.proc.is_alive() and self.queue:
            plotlist, step = self.queue[0]

            for i, p in enumerate(plotlist):
                if isinstance(p, PlotAsync):
                    res = p.get(wait)
                    if res is not None:
                        plotlist[i] = res
                    else:
                        if wait:
                            assert p.failed
                            # Exception in the worker thread
                            print("Exception detected in a PlotAsync object. Syncing logger and ignoring further plots.")
                            self.wait_logger(True)
                            self.request_stop()
                            self.proc.join()

                        return

            self.queue.pop(0)
            self.enqueue(plotlist, step)

        self.wait_logger(wait)

    def finish(self):
        if self.stop_event.is_set():
            return

        self.flush(True)
        self.request_stop()
        self.proc.join()
