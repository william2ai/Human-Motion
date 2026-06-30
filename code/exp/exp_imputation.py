from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import importlib
from ptflops import get_model_complexity_info
import pdb
import re
warnings.filterwarnings('ignore')


# 内存消耗
# dmesg | grep -i 3636466

 

def create_masks(batch_size, seq_len, num_sensors, mask_rate, mask_type, device):
    B = batch_size
    T = seq_len
    N = num_sensors

    # Initialize masks
    mask = torch.ones((B, T, N), device=device)  # Default to all unmasked

    if mask_type in (0, 2):  # Point masking for mask 0 and combined mode
        point_mask = (torch.rand((B, T, N), device=device) > mask_rate).float()
        mask *= point_mask  # Update mask with point masking

    if mask_type in (1, 2):  # Block masking
        block_mask = torch.ones((B, T, N), device=device)  # Start with a mask of ones
        
        # Create a random mask for block masking
        block_mask_random = torch.rand(B, N, device=device) < (mask_rate * 2)
        
        for n in range(N):
            # Generate random start indices for each sensor
            start_indices = torch.randint(0, T - 10 + 1, (B,), device=device)
            for b in range(B):
                if block_mask_random[b, n]:
                    start = start_indices[b]
                    block_mask[b, start:start + 10, n] = 0  # Set block to masked
        
        mask *= block_mask  # Combine with the existing mask

    return mask
    
def visualize(true, filled, mask, save_path):
    plt.figure(figsize=(12, 6))

    # Plot true values
    plt.plot(true, label='True Values', color='blue', alpha=0.5)

    # Plot filled values
    plt.plot(filled, label='Filled Values', color='orange', alpha=0.8)

    # Overlay masked regions
    # pdb.set_trace()
    masked_indices = np.where(mask == 0)[0]
    # print(masked_indices)
    
    if len(masked_indices) > 0:
        # Find continuous segments of masked indices
        segments = np.split(masked_indices, np.where(np.diff(masked_indices) != 1)[0] + 1)
        # segments=0
        
        for segment in segments:
            if len(segment) > 1:
                # Draw a line for continuous segments
                plt.plot(segment, true[segment], color='lightcoral', linewidth=4, alpha=0.5)
            else:
                # Draw individual points for isolated masked indices
                plt.plot(segment, true[segment], marker='o', color='lightcoral', markersize=5)


    plt.title('True vs Filled Values with Masked Regions', fontsize=20)  # Increase title font size
    plt.xlabel('Time Step', fontsize=18)  # Increase x-axis label font size
    plt.ylabel('Values', fontsize=20)  # Increase y-axis label font size
    plt.legend(fontsize=20)  # Increase legend font size
    plt.tick_params(axis='both', labelsize=18) 
    # plt.grid()
    # plt.tick_params(axis='both', which='major', size=15)  # Major ticks
    # plt.tick_params(axis='both', which='minor', size=15)  
    
     
    plt.savefig(save_path)
    plt.close()
    
class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
    
    def get_model_size(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # Convert to MB
    

    def _build_model(self):
        # pdb.set_trace()
        # Initialize the model
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag, distributed=None):
        data_set, data_loader = data_provider(self.args, flag, distributed=distributed)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0.0
        total_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,labels) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                B, T, N = batch_x.shape

                
                mask = create_masks(B, T, N, self.args.mask_rate, self.args.mask,batch_x.device)

                
                # mask[:, :, -1] = 0
                inp = batch_x.masked_fill(mask == 0, 0)
                    
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                masked = mask == 0
                if masked.any():
                    loss = criterion(outputs[masked], batch_x[masked])
                    total_loss += loss.item()
                    total_count += 1

        if getattr(self.args, 'distributed', False):
            stats = torch.tensor([total_loss, total_count], device=self.device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = stats[0].item()
            total_count = int(stats[1].item())

        total_loss = total_loss / max(total_count, 1)
        self.model.train()
        return total_loss



    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if self.is_main_process:
            os.makedirs(path, exist_ok=True)
        self.barrier()

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self.is_main_process)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=5, factor=0.5)

        
        # At the beginning of your training loop
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        # After the forward pass

        
        
        for epoch in range(self.args.train_epochs):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            iter_count = 0
            train_loss = []
            
            model_size_mb = self.get_model_size(self.model)
            print(f"Model Size: {model_size_mb:.2f} MB")

            self.model.train()
            epoch_time = time.time()

            
            for i, input in enumerate(train_loader):
                # print(len(input))
                (batch_x, batch_y, batch_x_mark, batch_y_mark,labels)=input
                # pdb.set_trace()
                

                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                B, T, N = batch_x.shape


                # Create an initial mask
                if self.args.mask in (0, 2):  # Point masking for mask 0 and combined mode
                    point_mask = torch.rand((B, T, N)).to(self.device)
                    point_mask[point_mask <= self.args.mask_rate] = 0  # Masked points
                    point_mask[point_mask > self.args.mask_rate] = 1   # Unmasked points
                    mask=point_mask
                else:
                    point_mask = torch.ones((B, T, N)).to(self.device)  # No point masking for mask 1
                    
                    mask=point_mask

                # Block masking
                block_mask=None
                if self.args.mask in (1, 2):
                    block_mask = torch.ones((B, T, N)).to(self.device)  # Start with a mask of ones
                    block_size = 10  # Size of the block to mask
                    
                    for b in range(B):
                        for n in range(N):
                            # Decide whether to mask this block based on masked_rate
                            if np.random.rand() < self.args.mask_rate*2:
                                # pdb.set_trace()
                                # Randomly choose a starting point for the block
                                start = np.random.randint(0, T - block_size + 1)
                                end = start + block_size

                                # Apply the mask to the selected block
                                block_mask[b, start:end, n] = 0  # Set block to masked
                    mask=block_mask

                # Combine point mask and block mask
                if self.args.mask==2:
                    mask = point_mask * block_mask  # Multiply to combine masks
                
                #### ADD
                # mask[:, :, -1] = 0
                
                inp = batch_x.masked_fill(mask == 0, 0)
      
                outputs = self.model(inp, batch_x_mark, None, None, mask)
                
                # Adjust dimensions based on features
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

    
                # pdb.set_trace()
                # Create a weight mask for loss calculation
                weight_mask = torch.ones_like(outputs)  # Initialize weight mask with ones
                
                if block_mask is not None:
                # Set weights: higher for block_mask, lower for point_mask
                    weight_mask[block_mask == 0] = 10.0  # Higher weight for blocked areas
                    weight_mask[(point_mask == 0) & (block_mask == 1)] = 1.0  # Lower weight for point_mask areas
                
                # # Calculate loss with weights
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                
                
                weighted_loss = (loss * weight_mask[mask == 0]).mean()  # Apply weights to loss

                train_loss.append(weighted_loss.item())

                if (i + 1) % 500 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = self.reduce_mean(np.mean(train_loss))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            # Step the scheduler
            scheduler.step(vali_loss)

            if self.is_main_process:
                early_stopping(vali_loss, self.unwrap_model(), path)
                early_stop = early_stopping.early_stop
            else:
                early_stop = False
            early_stop = self.broadcast_bool(early_stop)
            if early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.barrier()
        self.unwrap_model().load_state_dict(torch.load(best_model_path, map_location=self.device))
            # Visualization of the training loss
        if self.is_main_process:
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss, label='Training Loss', color='blue')
            plt.title('Training Loss over Time')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(path, 'training_loss.png')) 
            plt.close()

        return self.model
    
    
    def test(self, setting, test=0):
        if getattr(self.args, 'distributed', False):
            if not self.is_main_process:
                return

        test_data, test_loader = self._get_data(flag='test', distributed=False)
        model = self.unwrap_model()
        if test:
            print('loading model')
            print("Layer parameters before loading:")
            for name, param in model.named_parameters():
                print(name)
            model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device), strict=True)
            print("Layer parameters after loading:")
            for name, param in model.named_parameters():
                print(name)


        preds = []
        trues = []
        masks = [] 
        labels_out=[]
        masked_outputs=[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # pdb.set_trace()
        parts = folder_path.split('_')
        mask_index = parts.index('mask')
        mode = parts[mask_index - 1] 
        mask_rate=parts[mask_index+1]
        
        if 'Realdisp' in folder_path:
            name='Realdisp'
        elif 'Daphnet' in folder_path:
            name='Daphnet'
        model.eval()
        
        
        # # out=[]
        # print("Memory Summary Before Testing:")
        # print(torch.cuda.memory_summary())
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _,labels) in enumerate(test_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                B, T, N = batch_x.shape
                
                mask = create_masks(B, T, N, self.args.mask_rate, self.args.mask,batch_x.device)
                # print("Memory after mask:")
                # print(torch.cuda.memory_summary())
                
                # 测试某个通道
                # batch_x[:, :, -1] = 0  # Set the first column to 0
                # mask[:, :, -1] = 0  
                
                inp = batch_x.masked_fill(mask == 0, 0)

                
                outputs = model(inp, batch_x_mark, None, None, mask)
                # print("Memory after model:")
                # print(torch.cuda.memory_summary())
                if torch.isnan(outputs).any():
                    print("NaN detected in outputs")

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS 
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                labels_out.append(labels.detach().cpu().numpy())
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())
                masked_outputs.append(inp.detach().cpu())
                
                del batch_x, batch_x_mark, 

                # pdb.set_trace()
            #     if i % 20 == 0:
            #         filled = true[0, :, -1].copy()
            #         filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
            #  pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
    
            #         # Call the modified visual function
            #         visualize(true[0, :, -1], filled, mask[0, :, -1].detach().cpu().numpy(), os.path.join(folder_path, str(i) + '.pdf'))

        # pdb.set_trace()
        labels_out = np.concatenate(labels_out, 0) # (466624, 96)
        preds = np.concatenate(preds, 0) # (466624, 96, 9)
        trues = np.concatenate(trues, 0) # (466624, 96, 9)
        masks = np.concatenate(masks, 0) # (466624, 96)
        masked_outputs = np.concatenate(masked_outputs, 0) # (466624, 96)
        
        
        # pdb.set_trace()
        classification_path = './classification/{}/{}_{}/'.format(name, mask_rate, mode)
        os.makedirs(classification_path, exist_ok=True)
        np.save(os.path.join(classification_path, '{}_labels.npy'.format(mask_rate)), labels_out)
        np.save(os.path.join(classification_path, '{}_preds.npy'.format(mask_rate)), preds)
        np.save(os.path.join(classification_path, '{}_masked.npy'.format(mask_rate)), masked_outputs)
        os.makedirs('./classification/{}/'.format(name), exist_ok=True)
        np.save('./classification/{}/trues.npy'.format(name), trues)


        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))

        # Save metrics to a text file
        with open(os.path.join(folder_path, "result_imputation.txt"), 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        # Save metrics as text
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write('mae: {}\n'.format(mae))
            f.write('mse: {}\n'.format(mse))
            f.write('rmse: {}\n'.format(rmse))
            f.write('mape: {}\n'.format(mape))
            f.write('mspe: {}\n'.format(mspe))

        # Save predictions and true values to text files
        # np.savetxt(os.path.join(folder_path, 'pred.txt'), preds, fmt='%.6f')  # Adjust format as needed
        # np.savetxt(os.path.join(folder_path, 'true.txt'), trues, fmt='%.6f')  # Adjust format as needed

        # mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        # print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_imputation.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return
