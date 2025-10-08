from datasets import load_dataset


def pprint(title, data):
    print(f"### {title} ###")
    print(data)
    print("-----------------")


dataset = load_dataset("wmt14", "de-en")
pprint("FULL DATASET OVERVIEW", dataset)

# view some training samples
sample = dataset["train"][1]
de = sample["translation"]["de"]
en = sample["translation"]["en"]
pprint("SAMPLE DATA", f"ORIGINAL: \n\t{de}\n\nTRANSLATED: \n\t{en}")
