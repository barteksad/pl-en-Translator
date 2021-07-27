class CFG:
    model_name = "Helsinki-NLP/opus-mt-pl-en"
    dataset_name = "europa_eac_tm"
    train_batch_size = 8
    valid_batch_size = 2
    epochs = 1 
    lr = 1e-5   
    max_len = 128