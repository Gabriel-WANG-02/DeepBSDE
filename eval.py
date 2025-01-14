"""DeepBSDE evaluation script"""
import os
from mindspore import context, load_checkpoint
from src.net import DeepBSDE, WithLossCell
from src.config import config
from src.equation import get_bsde
from src.eval_utils import apply_eval

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.ckpt_path = os.path.join(config.log_dir, "deepbsde_{}_end.ckpt".format(config.eqn_name))
    bsde = get_bsde(config)
    print('Begin to solve', config.eqn_name)
    net = DeepBSDE(config, bsde)
    net_with_loss = WithLossCell(net)
    load_checkpoint(config.ckpt_path, net=net_with_loss)
    eval_param = {"model": net_with_loss, "valid_data": bsde.sample(config.valid_size)}
    loss, y_init = apply_eval(eval_param)
    print("eval loss: {}, Y0: {}".format(loss, y_init))
