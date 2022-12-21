# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""callback function"""
import wandb, os
import numpy as np

from src.args import args

from mindspore.train.callback._callback import Callback, _handle_loss
from mindspore._checkparam import Validator

class LossMonitor(Callback):

    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        Validator.check_non_negative_int(per_print_times)
        self._per_print_times = per_print_times
        self._last_print_time = 0
        wandb.init(reinit = False)

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.RunContext`.
        """
        cb_params = run_context.original_args()

        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        loss = _handle_loss(cb_params.net_outputs)
        wandb.log({"loss": loss})

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("In epoch: {} step: {}, loss is NAN or INF, training process cannot continue, "
                             "terminating training.".format(cur_epoch_num, cur_step_in_epoch))

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" % (cur_epoch_num, cur_step_in_epoch, loss), flush=True)

    def on_train_epoch_end(self, run_context):

        cb_params = run_context.original_args()
        metrics = cb_params.get("metrics")
        if metrics:
            print("Eval result: epoch %d, metrics: %s" % (cb_params.cur_epoch_num, metrics))

class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, total_epochs, save_freq=50):
        wandb.init(
            ...
        )
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.total_epochs = total_epochs
        self.save_freq = save_freq
        self.best_acc = 0.

    def on_train_epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num > self.total_epochs * 0.9 or int(
                cb_params.cur_epoch_num - 1) % 10 == 0 or cb_params.cur_epoch_num < 10:
            cur_epoch_num = cb_params.cur_epoch_num
            result = self.model.eval(self.eval_dataset)
            if result["acc"] > self.best_acc:
                self.best_acc = result["acc"]
            wandb.log({"acc":result["acc"]})
            print("epoch: %s acc: %s, best acc is %s" %
                  (cb_params.cur_epoch_num, result["acc"], self.best_acc), flush=True)
            if args.run_modelarts:
                import moxing as mox
                if cur_epoch_num % self.save_freq == 0:
                    mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
            
