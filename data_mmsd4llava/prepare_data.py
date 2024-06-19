import json
import pickle
import torch
from utils import cos_sim
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score, accuracy_score


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

def select_topk(sample, data_features, features4select, text_features4select, image_features4select, indice2image_id, topk=1):
    sample_features = data_features[str(sample["image_id"])]
    sample_text_feature = torch.tensor(sample_features["text_feature"]).to(device)
    sample_image_feature = torch.tensor(sample_features["image_feature"]).to(device)
    sample_text_feature = sample_text_feature / sample_text_feature.norm(dim=1, keepdim=True)
    sample_image_feature = sample_image_feature / sample_image_feature.norm(dim=1, keepdim=True)
    text_features4select = text_features4select / text_features4select.norm(dim=1, keepdim=True)
    image_features4select = image_features4select / image_features4select.norm(dim=1, keepdim=True)
    
    text_sims = torch.mm(sample_text_feature, text_features4select.T)
    image_sims = torch.mm(sample_image_feature, image_features4select.T)
    # sims = (text_sims + image_sims) / 2
    labmda = 0.5
    sims = labmda * text_sims + (1-labmda) * image_sims
    top_k_values, top_k_indices = torch.topk(sims, topk+1, dim=1)
    top_k_values = top_k_values.squeeze().tolist()
    top_k_indices = top_k_indices.squeeze().tolist()
    demons = []
    
    for i, indice in enumerate(top_k_indices):
        demon_image_id = indice2image_id[indice]
        if demon_image_id == str(sample["image_id"]):
            continue
        demon = {
            "image_id":demon_image_id,
            "text":features4select[demon_image_id]["text"],
            "label":features4select[demon_image_id]["label"],
            "score":top_k_values[i]
        }
        # print("sample",sample)
        # print("demon",demon)
        demons.append(demon)

    return demons

sufffix = "mmsd_demon"


data_folder = "/MMSD2.0"
for dataset in ["train","valid","test"]:
    print(f"Processing {dataset} now !")
    data_path = f"{data_folder}/{dataset}.json"
    data_features = load_file(f"{data_folder}/{dataset}_features")
    features4select = load_file(f"{data_folder}/train_features")
    text_features4select = []
    image_features4select = []
    indice2image_id = {}
    for i, (image_id, features) in enumerate(features4select.items()):
        indice2image_id[i] = image_id
        text_features4select.append(torch.tensor(features["text_feature"]).to(device))
        image_features4select.append(torch.tensor(features["image_feature"]).to(device))
    
    text_features4select = torch.stack(text_features4select,dim=0).squeeze()
    image_features4select = torch.stack(image_features4select,dim=0).squeeze()

    data_dict = json.load(open(data_path, "r"))
    total_num = len(data_dict)
    mm_results = []
    demons_labels = []
    sample_golds = []
    for i, sample in enumerate(data_dict):
        # print(sample)
        demon = select_topk(sample, data_features, features4select, text_features4select, image_features4select, indice2image_id, topk=1)[0] # TODO: topk>1
        # print(demons)
        mm_out_sample = {
            "id": f"{i}",
            "image": f"{demon['image_id']}.jpg,{sample['image_id']}.jpg",
            "conversations": [
            {
                "from": "human",
                "value": f"<image>\n<image>\nHere is a demonstration:\n<{demon['text']}> and the first image, the label is \"{demon['label']}\". Based on the demonstration, please select the sarcasm label of the <{sample['text']}> and the second image from <0, 1>:"
            },
            {
                "from": "gpt",
                "value": f"{sample['label']}"
            },
        ]}
        demons_labels.append(demon['label'])
        sample_golds.append(sample['label'])
        # if i % 1000 == 0:
        #     print(f"processed {i}/{total_num}= {i/total_num}")
        mm_results.append(mm_out_sample)
    res = {}
    res['Acc'] = accuracy_score(demons_labels, sample_golds)
    res['F1_macro'] = f1_score(demons_labels, sample_golds, average='macro')
    res['P_macro'] = precision_score(demons_labels, sample_golds, average='macro')
    res['R_macro'] = recall_score(demons_labels, sample_golds, average='macro')   
    print(res)
    if "MMSD2.0" in data_path:
        if dataset == "train":
            with open(f"{dataset}_{sufffix}.json","w") as fw:
                fw.write(json.dumps(mm_results, ensure_ascii=False,indent=4))
        else:
            with open(f"{dataset}_{sufffix}.json","w") as fw:
                fw.write(json.dumps(mm_results, ensure_ascii=False,indent=4))
    else:
        with open(f"{dataset}_mmsd1_{sufffix}.json","w") as fw:
            fw.write(json.dumps(mm_results, ensure_ascii=False,indent=4))
    