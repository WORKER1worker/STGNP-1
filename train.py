import os
import shlex
import time
from options.train_options import TrainOptions
from options.val_options import Valptions
from data import create_dataset
from models import create_model
from utils.logger import Logger


if __name__ == '__main__':
    opt, model_config = TrainOptions().parse()   # get training options
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)
    if opt.enable_val:
        val_opt, _ = Valptions().parse()  # get validation options
        val_dataset = create_dataset(val_opt)  # create a validation dataset given opt.dataset_mode and other options
        dataset_size = len(val_dataset)  # get the number of samples in the dataset.
        print('The number of validation samples = %d' % dataset_size)

    model = create_model(opt, model_config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    best_metric = float('inf')  # best metric
    n_target_start = opt.num_train_target
    early_stop_trigger = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:   # display images on visdom and save images to
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

            iter_data_time = time.time()

        if opt.enable_val and epoch % val_opt.eval_epoch_freq == 0:
            model.eval()
            val_start_time = time.time()
            for i, data in enumerate(val_dataset):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.test()
                model.cache_results()  # store current batch results
            t_val = time.time() - val_start_time
            model.compute_metrics()
            metrics = model.get_current_metrics()
            visualizer.print_current_metrics(epoch, total_iters, metrics, t_val)

            if opt.save_best and best_metric > metrics['RMSE']:
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                best_metric = metrics['RMSE']
                model.save_networks('best')
                early_stop_trigger = 0
            else:
                early_stop_trigger += val_opt.eval_epoch_freq

            model.clear_cache()

            # check early stopping
            early_stopping_threshold = 5 if opt.dataset_mode == 'Water' else 20
            if early_stop_trigger >= early_stopping_threshold:
                if early_stop_trigger >= 500:
                    print('Trigger early stopping!')
                    break

            # check kl term (only for variational methods)
            if opt.model in ['hierarchical', 'snp', 'gsnp'] and epoch > 5 and not model.kl_flag:
                raise RuntimeError('Bad initialization, terminate the program')

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Curriculum Learning (Not utilized in the project)
        if opt.enable_curriculum and epoch >= opt.n_epochs_target and opt.num_train_target <= opt.num_train_target_end:
            next_num_target = n_target_start + int((epoch - opt.n_epochs_target) / opt.n_epochs_target_increase * opt.num_train_target_end + 1)
            if opt.num_train_target != next_num_target:
                opt.num_train_target = next_num_target
                print('upload the number of target to %d' % next_num_target)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        new_lr = model.update_learning_rate()  # update learning rates in the beginning of every epoc

    print('Run the evaluation.')
    with open(os.path.join(model.save_dir,'run_test.sh'), 'w') as f:
        f.write('source activate pytorch-py39\n')

        gpu_id = opt.gpu_ids[0] if len(opt.gpu_ids) > 0 else -1
        cmd_parts = [
            'python', 'test.py',
            '--model', str(opt.model),
            '--dataset_mode', str(opt.dataset_mode),
            '--pred_attr', str(opt.pred_attr),
            '--gpu_ids', str(gpu_id),
            '--config', str(opt.config),
            '--file_time', str(opt.file_time),
            '--epoch', 'best',
        ]

        # Keep test command aligned with training data split for SM experiments.
        passthrough_args = [
            'sm_location_path',
            'sm_data_path',
            'sm_test_nodes_path',
            'sm_holdout_nodes_path',
            'sm_holdout_station_id',
            'sm_holdout_lon',
            'sm_holdout_lat',
        ]
        for arg_name in passthrough_args:
            if hasattr(opt, arg_name):
                value = getattr(opt, arg_name)
                if value is None or str(value).strip() == '':
                    continue
                cmd_parts.extend([f'--{arg_name}', str(value)])

        if hasattr(opt, 'sm_eval_target_mode'):
            # Auto-eval after training should run on regular test targets by default.
            cmd_parts.extend(['--sm_eval_target_mode', 'test'])

        cmd = ' '.join(shlex.quote(x) for x in cmd_parts)
        f.write(cmd)

    os.system('chmod u+x '+ os.path.join(model.save_dir,'run_test.sh'))
    os.system(os.path.join(model.save_dir,'run_test.sh'))