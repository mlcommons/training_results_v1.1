
import json

def save_log_info(solver, config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['num_samples'] = 4195197692
    config['eval_num_samples'] = 89137319
    config['global_batch_size'] = solver.batchsize
    config['opt_base_learning_rate'] = solver.lr
    config['sgd_opt_base_learning_rate'] = solver.lr
    config['sgd_opt_learning_rate_decay_poly_power'] = solver.decay_power
    config['opt_learning_rate_warmup_steps'] = solver.warmup_steps
    config['opt_learning_rate_warmup_factor'] = 0.0
    config['lr_decay_start_steps'] = solver.decay_start
    config['sgd_opt_learning_rate_decay_steps'] = solver.decay_steps
    config['gradient_accumulation_steps'] = 1

    with open(config_file, 'w') as f:
        json.dump(config, f)
    # log_str.append('eval_samples:{}'.format(model.layers[0].eval_num_samples))