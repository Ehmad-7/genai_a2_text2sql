import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent / "starter"))

import torch
import torch.nn as nn
import sentencepiece as spm

from dataset import make_loader
from tokenizer import PAD_ID
from model.transformer import Transformer

D_MODEL,WARMUP=256,4000

def noam_lr(step,d_model=D_MODEL,warmup=WARMUP):
    if step<1:
        step=1
    return (d_model**-0.5)*min(step**-0.5,step*(warmup**-1.5))

def run_epoch(model,loader,criterion,device,optimizer=None,scheduler=None,max_batches=0):
    training=optimizer is not None
    model.train(training)
    total_loss,total_tokens=0.0,0
    with torch.set_grad_enabled(training):
        for i,(src,tgt) in enumerate(loader):
            if max_batches and i>=max_batches:
                break
            src,tgt=src.to(device),tgt.to(device)
            
            decoder_input=tgt[:, :-1]
            labels=tgt[:,1:]
            
            logits=model(src,decoder_input)
            
            logits_flat=logits.contiguous().view(-1,logits.size(-1))
            labels_flat=labels.contiguous().view(-1)
            loss=criterion(logits_flat,labels_flat)
            
            n_tokens=(labels_flat!=PAD_ID).sum()
            if training:
                optimizer.zero_grad()
                if n_tokens>0:
                    (loss/n_tokens).backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                
            total_loss+=loss.item()
            non_pad_tokens=(labels_flat!=PAD_ID).sum().item()
            total_tokens+=non_pad_tokens
            
    return total_loss/total_tokens if total_tokens>0 else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--overfit",action="store_true",help="repeat ONE batch (debug)")
    ap.add_argument("--max_batches",type=int,default=0,help="limit batches per epoch (debug)")
    args=ap.parse_args()
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ",device,torch.cuda.get_device_name(0) if device.type=="cuda" else "")
    
    os.makedirs(args.ckpt_dir,exist_ok=True)
    os.makedirs("results",exist_ok=True)
    
    sp=spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    train_dl=make_loader("starter/train_pairs.jsonl",sp,train=True,batch_size=args.batch_size)
    dev_dl=make_loader("starter/dev_pairs.jsonl",sp,train=False,batch_size=args.batch_size)
    
    model=Transformer(sp.get_piece_size()).to(device)
    
    criterion=nn.CrossEntropyLoss(ignore_index=PAD_ID,label_smoothing=0.1,reduction="sum")
    
    optimizer=torch.optim.Adam(model.parameters(),lr=1.0,betas=(0.9,0.98),eps=1e-9)
    
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda s: noam_lr(s+1))
    
    start_epoch,best_dev=1,float("inf")
    
    if args.resume:
        
        ckpt_path=os.path.join(args.ckpt_dir,"last.pt")
        
        if os.path.exists(ckpt_path):
            print(f"Resuming training from checkpoint: {ckpt_path}")
            checkpoint=torch.load(ckpt_path,map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch=checkpoint["epoch"]+1
            best_dev=checkpoint["best_dev"]
        else:
            print(f"warning: checkpoint {ckpt_path} not found.starting from scratch.")
            
    if args.overfit:
        print("Overfitting debug mode activated. Training on a single batch for 200 steps.")
        
        one_batch=next(iter(train_dl))
        src,tgt=one_batch
        src,tgt=src.to(device),tgt.to(device)
        
        decoder_input=tgt[:, :-1]
        labels=tgt[:, 1:]
        labels_flat=labels.contiguous().view(-1)
        non_pad_tokens=(labels_flat!=PAD_ID).sum().item()
        
        for g in optimizer.param_groups:
            g["lr"]=1e-3
            
        model.train()
        
        for step in range(1,201):
            optimizer.zero_grad()
            logits=model(src,decoder_input)
            logits_flat=logits.contiguous().view(-1,logits.size(-1))
            loss=criterion(logits_flat,labels_flat)
            (loss/non_pad_tokens).backward()
            optimizer.step()
            
            if step % 20==0:
                mean_token_loss=loss.item()/non_pad_tokens if non_pad_tokens>0 else 0.0
                print(f"Step {step:3d} | Overfit loss per token: {mean_token_loss:.4f}")
                
        return 
    
    log_path=os.path.join(args.ckpt_dir,"train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","dev_loss","lr","seconds"])
            
    for epoch in range(start_epoch,args.epochs+1):
        t0=time.time()
        train_loss=run_epoch(model,train_dl,criterion,device,optimizer,scheduler,args.max_batches)
        dev_loss=run_epoch(model,dev_dl,criterion,device,max_batches=args.max_batches)
        lr=optimizer.param_groups[0]["lr"]
        secs=time.time()-t0
        
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | dev {dev_loss:.4f}"
              f"| lr {lr:.2e} | {secs:.0f}s")
        
        with open(log_path,"a",newline="") as f:
            csv.writer(f).writerow([epoch,train_loss,dev_loss,lr,secs])
            
        if dev_loss<best_dev:
            best_dev=dev_loss
            torch.save({
                "model_state_dict":model.state_dict(),
                "epoch":epoch,
                "dev_loss":dev_loss                
            },os.path.join(args.ckpt_dir,"best.pt"))
            
        torch.save({
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),
            "epoch":epoch,
            "best_dev":best_dev
        },os.path.join(args.ckpt_dir,"last.pt"))
        
if __name__=="__main__":
    main()
            
        
    
    