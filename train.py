import torch
from torch import nn
from src.models import MultiModalTransformerForClassification
from utils.util import *
import transformers
import time
from utils.eval_metrics import eval_meld
from pytorch_lightning.lite import LightningLite
from time import strftime
from utils.dataset import get_cluster_reps, cluster, gen_cl_data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from spcl_loss import SupProtoConLoss


# def get_each_label_count(datasets, class_num=7):
#     loss_rate = []
#     loss_rate_sum = 0
#     for i in range(class_num):
#         loss_rate.append(0)
#     for dataset in datasets:
#         for i in dataset.labels:
#             loss_rate[i] += 1
#     for i in range(class_num):
#         loss_rate[i] = 1 / loss_rate[i]
#         loss_rate_sum += loss_rate[i]
#     for i in range(class_num):
#         loss_rate[i] = loss_rate[i] / loss_rate_sum
#     return loss_rate


class FusionLoss(nn.Module):
    def __init__(self, ):
        super(FusionLoss, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, all_loss, text_loss, audio_loss):
        return self.a * all_loss + self.b * text_loss + self.c * audio_loss


class Lite(LightningLite):
    def run(self, args, trg_train_loader, trg_valid_loader, trg_test_loader):
        #-----------------------------------------------------define training and evaluation on main task-------------------------------------------------------#
        
        def multimodal_train(self, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion):
            multimodal_model.train()
            num_batches = args.trg_n_train // args.trg_batch_size      #num_of_utt // batch_size == len(trg_train_loader)
            total_loss, total_size = 0, 0
            start_time = time.time()
            multimodal_model_optimizer.zero_grad()

            for i_batch, batch in enumerate(trg_train_loader):
                batch_size = args.trg_batch_size
                loss = 0
                batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batch_label_ids, batchUtt_in_dia_idx = batch
                logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batchUtt_in_dia_idx)
                loss = criterion(logits, batch_label_ids.to('cuda'))
                loss = loss / args.trg_accumulation_steps

                self.backward(loss)

                if ((i_batch+1)%args.trg_accumulation_steps)==0:
                    torch.nn.utils.clip_grad_norm_(multimodal_model.parameters(), args.clip)
                    multimodal_model_optimizer.step()
                    multimodal_model_scheduler.step()
                    multimodal_model_optimizer.zero_grad()
                total_loss += loss.item() * batch_size * args.trg_accumulation_steps
                total_size += batch_size
                if i_batch % args.trg_log_interval == 0 and i_batch > 0:
                    avg_loss = total_loss / total_size
                    elapsed_time = time.time() - start_time
                    print('**TRG** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / args.trg_log_interval, avg_loss))
                    total_loss, total_size = 0, 0
                    start_time = time.time()

        def multimodal_evaluate(multimodal_model, criterion, test=False):

            multimodal_model.eval()
            loader = trg_test_loader if test else trg_valid_loader
            total_loss = 0.0
            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch in enumerate(loader):
                    batch_size = args.trg_batch_size
                    batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batch_label_ids, batchUtt_in_dia_idx = batch

                    logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batchUtt_in_dia_idx)

                    total_loss += criterion(logits, batch_label_ids.cuda()).item() * batch_size
                    # Collect the results into dictionary
                    results.append(logits)  #
                    truths.append(batch_label_ids)
            avg_loss = total_loss / (args.trg_n_test if test else args.trg_n_valid)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        #-------------------------------------------------------------------------------------------------------------------------------#
        # criterion_new = nn.CrossEntropyLoss(weight=loss_rate.to('cuda'))
        criterion = nn.CrossEntropyLoss()
        #training from scratch
        if not args.doEval:
            trg_train_loader = self.setup_dataloaders(trg_train_loader, move_to_device=False)
            trg_valid_loader = self.setup_dataloaders(trg_valid_loader, move_to_device=False)
            trg_test_loader  = self.setup_dataloaders(trg_test_loader, move_to_device=False)

            #------------------------------------------------------------Loading various unimodal or multimodal models and optimizers-----------------------------------------------------------#
            if args.choice_modality == 'T+A':
                multimodal_model = MultiModalTransformerForClassification(args)
                multimodal_model_optimizer = transformers.AdamW(multimodal_model.parameters(), lr=args.trg_lr, weight_decay=args.weight_decay, no_deprecation_warning=True)
                multimodal_model, multimodal_model_optimizer = self.setup(multimodal_model, multimodal_model_optimizer)  # Scale your model / optimizers
                multimodal_total_training_steps = args.num_epochs * len(trg_train_loader) / args.trg_accumulation_steps
                multimodal_model_scheduler = transformers.get_linear_schedule_with_warmup(optimizer = multimodal_model_optimizer,
                                                                num_warmup_steps = int(multimodal_total_training_steps * args.warm_up),
                                                                num_training_steps = multimodal_total_training_steps)

            #---------------------------------------------------------------------Adjust and optimize model-----------------------------------------------------------------------#
            best_valid_f1 = 0
            best_model_time = 0
            for epoch in range(args.num_epochs+1):
                if args.choice_modality == 'T+A':
                    #-----------------------------------------------------target task--------------------------------------------------------#
                    start = time.time()
                    multimodal_train(self, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion)
                    val_loss, results, truths = multimodal_evaluate(multimodal_model, criterion, test=False)
                    end = time.time()
                    duration = end-start

                    val_wg_av_f1 = eval_meld(results, truths, test=False)
                    print("-"*50)
                    print('**TRG** | Epoch {:2d} | Time {:5.4f} hour | val_wg_av_f1 {:5.4f} '.format(epoch, duration/3600, val_wg_av_f1))
                    print("-"*50)

                    #save the best model on validation set
                    if val_wg_av_f1 > best_valid_f1:
                        current_time = strftime("%m-%d-%H-%M-%S")
                        second_multimodel_path = os.path.join(args.save_Model_path,'multimodal_model_{}_{}.pt'.format(args.choice_modality, best_model_time))
                        if os.path.exists(second_multimodel_path):
                            os.remove(second_multimodel_path)
                        save_Multimodal_model(multimodal_model,args, current_time)
                        best_valid_f1 = val_wg_av_f1
                        best_model_time = current_time

        #---------------------------------------------------------------------test----------------------------------------------------------------------------#
            #conduct evaluation on the best model
            print("&"*50)
            if args.choice_modality == 'T+A' :
                best_multi_model = load_Multimodal_model(args.choice_modality, args.save_Model_path, best_model_time)
                _, results, truths = multimodal_evaluate(best_multi_model, criterion, test=True)
            print('**TEST** | wg_av_f1 {:5.4f} '.format(eval_meld(results, truths, test=True)))
            print('\n')

        #conduct evaluation directly without training
        elif args.doEval:
            load_project_path = os.path.abspath(os.path.dirname(__file__))
            print("&"*50)
            if args.choice_modality == 'T+A' :
                best_multi_model = torch.load(os.path.join(load_project_path, 'pretrained_model', args.load_multimodal_path))
                _, results, truths = multimodal_evaluate(best_multi_model, criterion, test=True)
            print('**TEST** | wg_av_f1 {:5.4f} '.format(eval_meld(results, truths, test=True)))
            print('\n')
