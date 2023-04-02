
import torch
import numpy as np
from tqdm import tqdm, trange
import os
import sys
import logging
from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mlc import MLC_ROOT
from mlc.data_loaders.mlc_mix_dataloader import MLC_MixedDataDataLoader
from mlc.data_loaders.mlc_simple_dataloader import MLC_SimpleDataLoader
from mlc.data_loaders.mlc_simple_dataloader import ListLayout
from mlc.utils.info_utils import print_run_information
from mlc.utils.io_utils import create_directory, save_json_dict
from mlc.utils.layout_utils import filter_out_noisy_layouts
from mlc.utils.loss_and_eval_utils import *
from collections import OrderedDict
from mlc.config.cfg import save_cfg
from mlc.datasets.utils import load_mvl_dataset
from mlc.scale_recover.scale_recover import ScaleRecover
from mlc.utils.entropy_mcl_utils import eval_entropy_from_boundaries
import json


class WrapperHorizonNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_paths()
        from mlc.models.HorizonNet.model import HorizonNet
        from mlc.models.HorizonNet.misc import utils as hn_utils
        from mlc.models.HorizonNet.dataset import visualize_a_data
        self.device = torch.device(
            f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else 'cpu')

        # Loaded trained model
        assert os.path.isfile(cfg.model.ckpt), f"ckpt does not exist"
        logging.info("Loading HorizonNet...")
        self.net = hn_utils.load_trained_model(
            HorizonNet, cfg.model.ckpt).to(self.device)
        logging.info(f"ckpt: {cfg.model.ckpt}")
        logging.info("HorizonNet Wrapper Successfully initialized")

        self.current_epoch = 0

    @staticmethod
    def set_paths():
        hn_dir = os.path.join(MLC_ROOT, "models", "HorizonNet")
        if hn_dir not in sys.path:
            sys.path.append(hn_dir)

    def infer_from_single_mage(self, image):
        img_ori = image[..., :3].transpose([2, 0, 1]).copy()
        # x = torch.FloatTensor([img_ori / img_ori.max()])
        x = torch.FloatTensor(np.array(img_ori / img_ori.max())[None, ])
        self.net.eval()
        with torch.no_grad():
            y_bon_, y_cor_ = self.net(x.to(self.device))
        data = np.vstack((y_bon_.cpu()[0], y_cor_.cpu()[0]))
        return data

    def estimate_within_list_ly(self, list_ly):
        """
        Estimates bon for all ly defined in a mvl scene described by the passed list_ly
        """
        layout_dataloader = DataLoader(
            ListLayout(list_ly),
            batch_size=self.cfg.runners.mvl.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed(),
        )
        self.net.eval()
        evaluated_data = {}
        for x in tqdm(layout_dataloader, desc="Estimating list_ly..."):
            with torch.no_grad():
                y_bon_, y_cor_ = self.net(x['images'].to(self.device))
            for y_, cor_, idx in zip(y_bon_.cpu(), y_cor_.cpu(), x['idx']):
                data = np.vstack((y_, cor_))
                evaluated_data[idx] = data

        [ly.recompute_data(phi_coord=evaluated_data[ly.idx]) for ly in list_ly]

    def train_loop(self):
        if not self.is_training:
            logging.warning("Wrapper is not ready for training")
            return False

        # ! Freezing some layer
        if self.cfg.model.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = self.net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(self.cfg.model.freeze_earlier_blocks + 1):
                logging.warn('Freeze block %d' % i)
                for m in blocks[i]:
                    for param in m.parameters():
                        param.requires_grad = False

        if self.cfg.model.bn_momentum != 0:
            for m in self.net.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = self.cfg.model.bn_momentum

        print_run_information(self.cfg)

        self.net.train()

        iterator_train = iter(self.train_loader)
        for _ in trange(len(self.train_loader),
                        desc=f"Training HorizonNet epoch:{self.current_epoch}/{self.cfg.model.epochs}"):

            self.iterations += 1
            x, y_bon_ref, std = next(iterator_train)
            y_bon_est, _ = self.net(x.to(self.device))

            if y_bon_est is np.nan:
                raise ValueError("Nan value")

            if self.cfg.model.loss == "L1":
                loss = compute_L1_loss(y_bon_est.to(
                    self.device), y_bon_ref.to(self.device))
            elif self.cfg.model.loss == "weighted_L1":
                loss = compute_weighted_L1(y_bon_est.to(
                    self.device), y_bon_ref.to(self.device), std.to(self.device), self.cfg.model.min_std)
            else:
                raise ValueError("Loss function no defined in config file")
            if loss.item() is np.NAN:
                raise ValueError("something is wrong")
            self.tb_writer.add_scalar(
                "train/loss", loss.item(), self.iterations)
            self.tb_writer.add_scalar(
                "train/lr", self.lr_scheduler.get_last_lr()[0], self.iterations)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), 3.0, norm_type="inf")
            self.optimizer.step()

        self.lr_scheduler.step()

        # Epoch finished
        self.current_epoch += 1

        # ! Saving model
        if self.cfg.model.get("save_every") > 0:
            if self.current_epoch % self.cfg.model.get("save_every", 5) == 0:
                self.save_model(f"model_at_{self.current_epoch}.pth")

        if self.current_epoch > self.cfg.model.epochs:
            self.is_training = False

        # # ! Saving current epoch data
        # fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        # save_json_dict(filename=fn, dict_data=self.curr_scores)

        return self.is_training

    def save_current_scores(self):
        # ! Saving current epoch data
        fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        save_json_dict(filename=fn, dict_data=self.curr_scores)
        # ! Save the best scores in a json file regardless of saving the model or not
        save_json_dict(
            dict_data=self.best_scores,
            filename=os.path.join(self.dir_ckpt, "best_score.json")
        )


    # ! METHODS FOR VALIDATION
    def valid_iou_loop(self, only_val=False):
        print_run_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        total_eval = {}
        invalid_cnt = 0

        for _ in trange(len(iterator_valid_iou), desc="IoU Validation epoch %d" % self.current_epoch):
            x, y_bon_ref, std = next(iterator_valid_iou)

            with torch.no_grad():
                y_bon_est, _ = self.net(x.to(self.device))

                true_eval = {"2DIoU": [], "3DIoU": []}
                for gt, est in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy()):
                    eval_2d3d_iuo(est[None], gt[None], true_eval)

                local_eval = dict(
                    loss=compute_weighted_L1(y_bon_est.to(
                        self.device), y_bon_ref.to(self.device), std.to(self.device))
                )
                local_eval["2DIoU"] = torch.FloatTensor(
                    [true_eval["2DIoU"]]).mean()
                local_eval["3DIoU"] = torch.FloatTensor(
                    [true_eval["3DIoU"]]).mean()
            try:
                for k, v in local_eval.items():
                    if v.isnan():
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item() * x.size(0)
            except:
                invalid_cnt += 1
                pass

        if only_val:
            scaler_value = self.cfg.runners.valid_iou.batch_size * \
                (len(iterator_valid_iou) - invalid_cnt)
            curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
            curr_score_2d_iou = total_eval["2DIoU"] / scaler_value
            logging.info(f"3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"2D-IoU score: {curr_score_2d_iou:.4f}")
            return {"2D-IoU": curr_score_2d_iou, "3D-IoU": curr_score_3d_iou}

        scaler_value = self.cfg.runners.valid_iou.batch_size * \
            (len(iterator_valid_iou) - invalid_cnt)
        for k, v in total_eval.items():
            k = "valid_IoU/%s" % k
            self.tb_writer.add_scalar(
                k, v / scaler_value, self.current_epoch)

        # Save best validation loss model
        curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
        curr_score_2d_iou = total_eval["2DIoU"] / scaler_value

        # ! Saving current score
        self.curr_scores['iou_valid_scores'] = dict(
            best_3d_iou_score=curr_score_3d_iou,
            best_2d_iou_score=curr_score_2d_iou
        )

        if self.best_scores.get("best_iou_valid_score") is None:
            logging.info(f"Best 3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"Best 2D-IoU score: {curr_score_2d_iou:.4f}")
            self.best_scores["best_iou_valid_score"] = dict(
                best_3d_iou_score=curr_score_3d_iou,
                best_2d_iou_score=curr_score_2d_iou
            )
        else:
            best_3d_iou_score = self.best_scores["best_iou_valid_score"]['best_3d_iou_score']
            best_2d_iou_score = self.best_scores["best_iou_valid_score"]['best_2d_iou_score']

            logging.info(
                f"3D-IoU: Best: {best_3d_iou_score:.4f} vs Curr:{curr_score_3d_iou:.4f}")
            logging.info(
                f"2D-IoU: Best: {best_2d_iou_score:.4f} vs Curr:{curr_score_2d_iou:.4f}")

            if best_3d_iou_score < curr_score_3d_iou:
                logging.info(
                    f"New 3D-IoU Best Score {curr_score_3d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_3d_iou_score'] = curr_score_3d_iou
                self.save_model("best_3d_iou_valid.pth")

            if best_2d_iou_score < curr_score_2d_iou:
                logging.info(
                    f"New 2D-IoU Best Score {curr_score_2d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_2d_iou_score'] = curr_score_2d_iou
                self.save_model("best_2d_iou_valid.pth")

    def valid_h_loop(self, only_val=False):

        dataset = load_mvl_dataset(self.cfg)
        dataset.load_imgs = True
        dataset.load_gt_labels = False
        dataset.load_npy = False

        scale_recover = ScaleRecover(self.cfg)

        entropy_data_values = []
        for scene in tqdm(dataset.list_scenes, desc="Reading MVL scenes..."):
            logging.info(f"Scene Name: {scene}")
            print_run_information(self.cfg)

            list_ly = dataset.get_list_ly(scene_name=scene)

            # ! Overwrite phi_coord by the estimates
            self.estimate_within_list_ly(list_ly)
            if self.cfg.runners.mvl.apply_scale_recover:
                scale_recover.fully_vo_scale_estimation(list_ly=list_ly)

            print_run_information(self.cfg)  # we need it because VO-SCALE recover print several lines
        
            entropy_data = eval_entropy_from_boundaries(
                list_boundaries=[ly.boundary_floor for ly in list_ly],
                grid_size=self.cfg.runners.mvl.grid_size,
                min_likelihood=self.cfg.runners.mvl.min_likelihood_percent,
                padding=self.cfg.runners.mvl.padding,
                xedges=None,
                zedges=None
            )

            entropy_data_values.append(entropy_data['entropy'])
            logging.info(f"Entropy MEAN score: {np.mean(entropy_data_values):.4f}")
            logging.info(f"Entropy MED score: {np.median(entropy_data_values):.4f}")
            logging.info(f"Entropy MAX score: {np.max(entropy_data_values):.4f}")
            logging.info(f"Entropy MIN score: {np.min(entropy_data_values):.4f}")
            logging.info(f"Entropy STD score: {np.std(entropy_data_values):.4f}")

        if only_val:
            return

        self.tb_writer.add_scalar(
            "valid_H/mean", np.mean(entropy_data_values), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_H/median", np.median(entropy_data_values), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_H/max", np.max(entropy_data_values), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_H/min", np.min(entropy_data_values), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_H/std", np.std(entropy_data_values), self.current_epoch)

        #! Current Entropy val
        curr_h = np.mean(entropy_data_values)

        if self.best_scores.get("best_h_valid_score") is None:
            logging.info(f"Best H score: {curr_h:.4f}")
            self.best_scores["best_h_valid_score"] = curr_h
            self.best_scores["best_mse_h_valid_score"] = self.curr_scores["mean_mse_H"]
        else:
            best_h_score = self.best_scores["best_h_valid_score"]

            logging.info(
                f"H: Best: {best_h_score:.4f} vs Curr:{curr_h:.4f}")

            if best_h_score > curr_h:
                logging.info(
                    f"New H Best Score {curr_h: 0.4f}")
                self.best_scores["best_h_valid_score"] = curr_h
                self.best_scores["best_mse_h_valid_score"] = self.curr_scores["mean_mse_H"]
                self.save_model("best_h_valid.pth")

    def save_model(self, filename):
        if not self.cfg.model.get("save_ckpt", True):
            return

        # ! Saving the current model
        state_dict = OrderedDict(
            {
                "args": self.cfg,
                "kwargs": {
                    "backbone": self.net.backbone,
                    "use_rnn": self.net.use_rnn,
                },
                "state_dict": self.net.state_dict(),
            }
        )
        torch.save(state_dict, os.path.join(
            self.dir_ckpt, filename))

    def prepare_for_training(self):
        self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        self.set_optimizer()
        self.set_scheduler()
        self.set_train_dataloader()
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'cfg.yaml'), self.cfg)

    def set_log_dir(self):
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        create_directory(output_dir, delete_prev=False)
        logging.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.dir_log)

    def set_train_dataloader(self):
        logging.info("Setting Training Dataloader")
        if self.cfg.runners.train.get("mix_data_dir", None) is not None:    
            if self.cfg.runners.train.mix_data_dir.active:
                self.train_loader = DataLoader(
                    MLC_MixedDataDataLoader(self.cfg.runners.train),
                    batch_size=self.cfg.runners.train.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.cfg.runners.train.num_workers,
                    pin_memory=True if self.device != 'cpu' else False,
                    worker_init_fn=lambda x: np.random.seed(),
                )
                return
                
        # ! By default it will train as self_supervised (no GT data mixed)
        self.train_loader = DataLoader(
            MLC_SimpleDataLoader(self.cfg.runners.train),
            batch_size=self.cfg.runners.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.runners.train.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(),
        )

    def set_valid_dataloader(self):
        logging.info("Setting IoU Validation Dataloader")
        self.valid_iou_loader = DataLoader(
            MLC_SimpleDataLoader(self.cfg.runners.valid_iou),
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def set_optimizer(self):
        if self.cfg.model.optimizer == "SGD":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                momentum=self.cfg.model.beta1,
                weight_decay=self.cfg.model.weight_decay,
            )
        elif self.cfg.model.optimizer == "Adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                betas=(self.cfg.model.beta1, 0.999),
                weight_decay=self.cfg.model.weight_decay,
            )
        else:
            raise NotImplementedError()

    def set_scheduler(self):
        decayRate = self.cfg.model.lr_decay_rate
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )
