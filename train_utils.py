from models import main_model
from torch import optim
import time
import numpy as np
import logging
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

def _train_epoch(model, train_loader, optimizer, epoch_no):
    """Execute one training epoch."""
    model.train()
    avg_loss = 0
    with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) in enumerate(it, start=1):
            optimizer.zero_grad()
            loss = model(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            it.set_postfix({"avg_loss": avg_loss / batch_no, "epoch": epoch_no}, refresh=False)
    return avg_loss / batch_no

def _validate_epoch(model, valid_loader, epoch_no):
    """Execute one validation epoch."""
    model.eval()
    avg_loss_valid = 0
    with torch.no_grad():
        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) in enumerate(it, start=1):
                loss = model(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask, is_train=0)
                avg_loss_valid += loss.item()
                it.set_postfix({"valid_avg_loss": avg_loss_valid / batch_no, "epoch": epoch_no}, refresh=False)
    return avg_loss_valid / batch_no

def diffusion_train(configs, train_loader, valid_loader=None):
    model = main_model.CDTI(configs).to(configs.device)
    optimizer = Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
    logging.info(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    lr_scheduler = MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    valid_epoch_interval = configs.valid_epoch_interval
    best_valid_loss = 1e10

    for epoch_no in range(configs.epoch_diff):
        epoch_start_time = time.time()
        
        train_loss = _train_epoch(model, train_loader, optimizer, epoch_no)
        logging.info(f"Epoch {epoch_no}: avg_loss={train_loss}")
        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            valid_loss = _validate_epoch(model, valid_loader, epoch_no)
            logging.info(f"Epoch {epoch_no}: valid_avg_loss={valid_loss}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print(f"Best loss updated to {valid_loss:.6f} at epoch {epoch_no}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch_no} finished in {epoch_duration:.2f} seconds.")
        logging.info(f"Epoch {epoch_no} finished in {epoch_duration:.2f} seconds.")

    return model

def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p])**2)
    return torch.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    return torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler=0, scaler=1):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = torch.sum(torch.abs(target * eval_points))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = [torch.quantile(forecast[j:j+1], quantiles[i], dim=1) for j in range(len(forecast))]
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return (CRPS / len(quantiles)).item()

def diffusion_test(configs, model, test_loader):
    model.eval()
    error_sum = 0
    missing_sum = 0
    generate_data2d = []
    all_metrics = {
        'target': [],
        'forecast': [],
        'eval_points': [],
        'all_target': [],
        'all_samples': [],
    }

    print(f"Test batches: {len(test_loader.dataset) // configs.batch + 1}")
    start = time.time()
    
    for i, (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) in enumerate(test_loader):
        imputed_samples, c_target, eval_points, observed_points, observed_time = model.evaluate(
            observed_data, observed_dataf, observed_mask, observed_tp, gt_mask
        )
        
        imputed_samples = imputed_samples.permute(0, 1, 3, 2)
        c_target = c_target.permute(0, 2, 1)
        eval_points = eval_points.permute(0, 2, 1)
        observed_points = observed_points.permute(0, 2, 1)
        
        for key in all_metrics:
            if key == 'target':
                all_metrics[key].append(c_target)
            elif key == 'forecast':
                all_metrics[key].append(observed_points)
            elif key == 'eval_points':
                all_metrics[key].append(eval_points)
            elif key == 'all_target':
                all_metrics[key].append(c_target)
            elif key == 'all_samples':
                all_metrics[key].append(imputed_samples)

        imputed_sample = imputed_samples.median(dim=1).values.detach().to("cpu")
        imputed_data = observed_mask * observed_data + (1 - observed_mask) * imputed_sample
        evalmask = gt_mask - observed_mask
        
        truth = observed_data * evalmask
        predict = imputed_data * evalmask
        error = torch.sum((truth - predict)**2)
        error_sum += error
        missing_sum += torch.sum(evalmask)
        
        B, L, K = imputed_data.shape
        generate_data2d.append(imputed_data.reshape(B*L, K).detach().to("cpu").numpy())
        
        print(f"Batch {i+1} time: {time.time() - start:.2f}s")
        start = time.time()
    
    generate_data2d = np.vstack(generate_data2d)
    np.savetxt("CDTI_Imputation.csv", generate_data2d, delimiter=",")
    
    target_2d = torch.cat(all_metrics['target'], dim=0)
    forecast_2d = torch.cat(all_metrics['forecast'], dim=0)
    eval_p_2d = torch.cat(all_metrics['eval_points'], dim=0)
    all_target = torch.cat(all_metrics['all_target'], dim=0)
    all_generated_samples = torch.cat(all_metrics['all_samples'], dim=0)
    
    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)
    MAE = calc_MAE(target_2d, forecast_2d, eval_p_2d)
    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, eval_p_2d)
    
    print(f"RMSE: {RMSE:.6f}")
    print(f"MAE: {MAE:.6f}")
    print(f"CRPS: {CRPS:.6f}")

