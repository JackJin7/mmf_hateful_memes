import pickle
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit, BertLayerJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.torchscript import getattr_torchscriptable
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
# from tsne_torch import TorchTSNE as TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


save_dir='/home/kkiriri/uoft/2516_project/mmf_hateful_memes/images/vilbert_uni'
db_dir='/home/kkiriri/uoft/2516_project/mmf_hateful_memes/images/vilbert_uni/my_vilbert_tsne_visualization_data.pkl'

def main():

    # t=torch.randint(low=0, high=2,size=(100,))
    # m=torch.rand(size=(100,768),dtype=float)
    # tsne_plot(t,m)
    data=[]
    concat_list=[]
    targets=[]

    with open(db_dir, 'rb') as f:
        while True:
            try:
                data=pickle.load(f)
                # take target as label for clustering
                inputs=data.get('input')
                targets.append(inputs.get('targets'))
                value=data.get('output')

                # take avg pooling from some of the all_hidden_states for TSNE
                # concat pooled_output for TSNE
                for embedding, layers in value.items():
                    if embedding=="pooled_output":
                        for i in layers:
                            concat_list.append(i)

                    # if embedding=='all_hidden_states':
                    #     hidden_depth=[12]
                    #     for i in hidden_depth:
                    #         layers[i].sum(axis=1)/((layers[i]> 0).sum(axis=1)).view(2,-1,768).compress()
                    #     pass

            except EOFError:
                print('Read finished')
                break

        avg_pooled=torch.stack(concat_list,dim=0)
        target=torch.cat(targets,dim=0)
        for i in range(10):
            tsne_plot(target,avg_pooled,i)




def tsne_plot(targets,input,i):

    # Apply PCA to reduce feature dimensions for fater computation and better tsne result, 30 is the compressed # dimension
    components=min([input.shape[1],input.shape[0],30])
    input=PCA(n_components=components,random_state=1).fit_transform(input)

    tsne = TSNE(n_components=2, perplexity=500, n_iter=10000, verbose=True,random_state=i)
    tsne_output = tsne.fit_transform(input)
    print(tsne_output.shape)
    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    print(targets)

    df['hatefulness'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='hatefulness',
        palette=sns.color_palette(n_colors=2),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5,
        s=15
    )
    # plt.legend(markerscale=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    filename='WithoutPCA_'+str(i)
    plt.savefig(os.path.join(save_dir,filename), bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    main()
