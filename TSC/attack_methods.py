import torch
import torch.nn as nn
import scipy.stats as st
import numpy as np
import torchvision
from PIL import Image
import random
import time
import math
import torch.nn.functional as F
from pick_restore import mask_video_frames,restore_masked_frames

def norm_grads(grads, frame_level=True):
    # frame level norm
    # clip level norm
    assert len(grads.shape) == 5 and grads.shape[2] == 32
    if frame_level:
        norm = torch.mean(torch.abs(grads), [1,3,4], keepdim=True)
    else:
        norm = torch.mean(torch.abs(grads), [1,2,3,4], keepdim=True)
    # norm = torch.norm(grads, dim=[1,2,3,4], p=1)
    return grads / norm

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
    
    def forward(self, outputs, targets):
       
        log_softmax_outputs = F.log_softmax(outputs, dim=1)
        
        if targets.dtype == torch.long:
            targets = F.one_hot(targets, num_classes=outputs.size(1)).float()
        softmax_targets = F.softmax(targets, dim=1)

        loss = F.kl_div(log_softmax_outputs, softmax_targets, reduction='batchmean')
        return loss

class Attack(object):
    """
    # refer to https://github.com/Harry24k/adversarial-attacks-pytorch
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._targeted = 1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._target_map_function = lambda images, labels:labels

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def set_attack_mode(self, mode, target_map_function=None):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'default' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
                         
            target_map_function (function) :
        """
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        if (mode == 'targeted') and (target_map_function is None):
            raise ValueError("Please give a target_map_function, e.g., lambda images, labels:(labels+1)%10.")
            
        if mode=="default":
            self._attack_mode = "default"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode=="targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._target_map_function = target_map_function
            self._transform_label = self._get_target_label
        elif mode=="least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(mode + " is not a valid mode. [Options : default, targeted, least_likely]")
            
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print('- Save Progress : %2.2f %% / Accuracy : %2.2f %%' % ((step+1)/total_batch*100, acc), end='\r')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print('\n- Save Complete!')

        self._switch_model()
    
    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=self.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=self.device)
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return video

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_" :
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default' :
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images

class Temporal_Skip_Connection(Attack):

    def __init__(self, model, params,mask_num,top_k,epsilon=16/255, steps=1, delay=1.0):
        super(Temporal_Skip_Connection, self).__init__("Temporal_Skip_Connection", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.delay = delay
        self.mask_num = mask_num
        self.top_k = top_k
        self.RKL_loss = KLDivergenceLoss()
        
        for name, value in params.items():
            setattr(self, name, value)
        
        self.frames = 32
        if self.kernel_mode == 'gaussian':
            kernel = self._initial_kernel_gaussian(self.kernlen).astype(np.float32)
        elif self.kernel_mode == 'linear':
            kernel = self._initial_kernel_linear(self.kernlen).astype(np.float32)   
        elif self.kernel_mode == 'uniform':
            kernel = self._initial_kernel_uniform(self.kernlen).astype(np.float32)  

        self.kernel = torch.from_numpy(np.expand_dims(kernel, 0)).to(self.device)

        ti_kernel = self._initial_kernel(15, 3).astype(np.float32)                           
        stack_kernel = np.stack([ti_kernel, ti_kernel, ti_kernel])                           
        self.stack_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(self.device)
    
    def _initial_kernel_linear(self, kernlen):
        k = int((kernlen - 1) / 2)
        kern1d = []
        for i in range(k+1):
            kern1d.append(1 - i / (k+1))
        kern1d = np.array(kern1d[::-1][:-1] + kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_uniform(self, kernlen):
        kern1d = np.ones(kernlen)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_gaussian(self, kernlen):
        assert kernlen%2 == 1
        k = (kernlen - 1) /2
        sigma = k/3
        k = int(k)
        def calculte_guassian(x, sigma):
            return (1/(sigma*np.sqrt(2*np.pi)) * np.math.exp(-(x**2)/(2* (sigma**2))))
        kern1d = []
        for i in range(-k, k+1):
            kern1d.append(calculte_guassian(i, sigma))
        assert len(kern1d) == kernlen
        kern1d = np.array(kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _conv1d_frame(self, grads):
        '''
        grads: D, N, C, T, H, W
        '''
        D,N,C,T,H,W = grads.shape
        grads = grads.reshape(D, -1)
        
        grad = torch.matmul(self.kernel, grads)
        grad = grad.reshape(N,C,T,H,W)
        return grad

    def _calculate_loss(self, outputs, loss1, loss2, true_labels, k):
        device = outputs.device  
        cost1 = loss1(outputs, true_labels)
        _,topk_labels = self._get_topk_label(outputs, true_labels, k)
        cost2 = 0
        for i in range(k):  
            target_distributions = F.one_hot(topk_labels[:, i], num_classes=outputs.size(1)).to(device).float()
            cost2 += loss2(outputs, target_distributions)
        cost2 /= k 
        cost = cost1 - cost2
        return cost
 
    def _get_grad(self, adv_videos, labels, loss1,loss2,top_k):
        batch_size = adv_videos.shape[0]
        used_labels = torch.cat([labels]*batch_size, dim=0)
        adv_videos.requires_grad = True                                         
        outputs = self.model(adv_videos)                                        
        cost = self._targeted*self._calculate_loss(outputs,loss1,loss2,used_labels,top_k)
        grad = torch.autograd.grad(cost, adv_videos, 
                                    retain_graph=False, create_graph=False)[0]
        return grad
    
    def _initial_kernel(self, kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _conv2d_frame(self, grads):
        '''
        grads: N, C, T, H, W
        '''
        frames = grads.shape[2]                                                                          
        out_grads = torch.zeros_like(grads)
        for i in range(frames):
            this_grads = grads[:,:,i]                                                                    
            out_grad = nn.functional.conv2d(this_grads, self.stack_kernel, groups=3, stride=1, padding=7)
            out_grads[:,:,i] = out_grad                                                                  
        out_grads = out_grads / torch.mean(torch.abs(out_grads), [1,2,3], True)                          
        return out_grads

    def _get_topk_label(self,outputs, true_labels, k):
        probabilities = F.softmax(outputs, dim=1)  

        for i, label in enumerate(true_labels):
            probabilities[i, label] = 0

        topk_probabilities, topk_indices = torch.topk(probabilities, k + 1, dim=1) 

        topk_filtered_indices = []
        topk_filtered_probabilities = []
        for i in range(len(true_labels)):
            mask = topk_indices[i] != true_labels[i]
            filtered_indices = topk_indices[i][mask][:k]
            filtered_probabilities = topk_probabilities[i][mask][:k]
            topk_filtered_indices.append(filtered_indices)
            topk_filtered_probabilities.append(filtered_probabilities)

        return torch.stack(topk_filtered_probabilities), torch.stack(topk_filtered_indices)

    def forward(self, videos, labels):

        videos = videos.to(self.device)  
        momentum = torch.zeros_like(videos).to(self.device)                                                        
        labels = labels.to(self.device)

        loss1 = nn.CrossEntropyLoss()   
        loss2 = self.RKL_loss
              
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') 
        adv_videos = videos.clone().detach()
        del videos

        start_time = time.time()
        for q in range(self.steps):
            batch_new_videos = []                                                                                                   
            batch_new_list = []                                                                                                     

            for i in range((self.kernlen-1)//2): 
                new_videos_A,list_A,new_videos_B,list_B = mask_video_frames(adv_videos,self.mask_num)
                batch_new_videos.append(new_videos_A)
                batch_new_videos.append(new_videos_B)
                batch_new_list.append(list_A)   
                batch_new_list.append(list_B)

         
            batch_new_videos.insert(self.kernlen//2,adv_videos)
            batch_inps = torch.cat(batch_new_videos, dim=0)                     

            grads = []
            batch_size = 1
            length = self.kernlen
            
            batch_times = self.kernlen
            for m in range(batch_times):
                grad = self._get_grad(batch_inps[m*batch_size:min((m+1)*batch_size,length)],labels,loss1,loss2,self.top_k)
                grad = self._conv2d_frame(grad)
                grads.append(grad)
 
            grads = torch.cat(grads,dim=0)
            grads = restore_masked_frames(grads,batch_new_list)

            grads = torch.unsqueeze(grads, dim=1)
            grad = self._conv1d_frame(grads)

            if self.momentum:
                grad = norm_grads(grad)
                grad += momentum * self.delay
                momentum = grad

            adv_videos = self._transform_video(adv_videos.detach(), mode='back')                 
            adv_videos = adv_videos + self.step_size*grad.sign()                                 
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()               
            adv_videos = self._transform_video(adv_videos, mode='forward')                       
            print ('now_time', time.time()-start_time)
        
        return adv_videos