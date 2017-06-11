import matplotlib.pyplot as plt
import os
import sys
import csv
from params import get_parser

def parseLossLog(filename,params):

    if params.nlosscurves == 1:
        train_losses = []
        val_losses = []
    else:
        train_losses = {'cos':[],'class_im':[],'class_rec':[],'total':[]}
        val_losses = {'cos':[],'class_im':[],'class_rec':[],'total':[]}

    totaltime = 0
    with open(filename, 'rU') as csvfile:

        lines = csv.reader( csvfile, delimiter='\n')

        for i,line in enumerate(lines):

            if len(line)>0: # if line is not empty

                line = line[0]

                if 'loss' in line and not 'val' in line and not 'Val' in line:
                    line = line.split('\t')
                    cos_loss = line[0]
                    if params.nlosscurves==1:
                        train_losses.append(float(cos_loss.split(': ')[1]))
                    else:
                        cls_loss1 = line[1]
                        cls_loss2 = line[2]
                        total = line[3]

                        train_losses['cos'].append(float(cos_loss.split(': ')[1]))
                        train_losses['class_im'].append(float(cls_loss1.split(': ')[1]))
                        train_losses['class_rec'].append(float(cls_loss2.split(': ')[1]))
                        train_losses['total'].append(float(total.split(': ')[1]))

                elif ('(val)' in line or 'Val' in line) and not 'valfreq' in line:
                    line = line.split('\t')
                    cos_loss = line[0]
                    if params.nlosscurves==1:
                        val_losses.append(float(cos_loss.split(': ')[1]))
                    else:
                        cls_loss1 = line[1]
                        cls_loss2 = line[2]
                        total = line[3]

                        val_losses['cos'].append(float(cos_loss.split(': ')[1]))
                        val_losses['class_im'].append(float(cls_loss1.split(': ')[1]))
                        val_losses['class_rec'].append(float(cls_loss2.split(': ')[1]))
                        val_losses['total'].append(float(total.split(': ')[1]))
                elif 'Time' in line:

                    time = line.split('Time:')[-1].split(' ')[0]
                    totaltime+=float(time)

        print "Running time:",totaltime,'seconds:'
        h = totaltime/3600
        m = totaltime%3600/60
        print int(h), 'hours and',int(m), 'minutes.'
        return train_losses,val_losses


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()

    filename = params.logfile

    train_losses,val_losses = parseLossLog(filename,params)
    fs = 9
    if params.nlosscurves==1:
        t_train = range(params.dispfreq,len(train_losses)*params.dispfreq+1,params.dispfreq)

        t_val = range(params.valfreq,len(val_losses)*params.valfreq+1,params.valfreq) # validation loss is displayed after each epoch only
        plt.plot(t_train,train_losses,'b-*',label='loss (t)')
        plt.plot(t_val,val_losses,'r-*',label='loss (v)')
    else:
        t_train = range(params.dispfreq,len(train_losses['cos'])*params.dispfreq+1,params.dispfreq)
        fig, axarr = plt.subplots(1,3)
        t_val = range(params.valfreq,len(val_losses['cos'])*params.valfreq+1,params.valfreq)

        axarr[0].plot(t_train,train_losses['cos'],'b-*',label='Cos loss (t)')
        axarr[0].plot(t_val,val_losses['cos'],'r-*',label='Cos loss (v)')
        #axarr[0].set_xticklabels(t_train, fontsize=fs,rotation=90)

        axarr[1].plot(t_train,train_losses['class_im'],'y-*',label='Cls-im loss (t)')
        axarr[1].plot(t_val,val_losses['class_im'],'g-*',label='Cls-im loss (v)')
        axarr[1].plot(t_train,train_losses['class_rec'],'b-*',label='Cls-rec loss (t)')
        axarr[1].plot(t_val,val_losses['class_rec'],'r-*',label='Cls-rec loss (v)')

        axarr[2].plot(t_train,train_losses['total'],'b-*',label='Total (t)')
        axarr[2].plot(t_val,val_losses['total'],'r-*',label='Total (v)')

        axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        axarr[2].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
