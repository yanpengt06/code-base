# Introduction

This repo includes my customized and frequently used code framework used for a wide range of NLP experiments and some awesome(I think) functions in tutils.py. **Updating continuously...**

The framework includes main function, dataset, dataloader, model definition, train/valide loop and so on.

The main structure refers to the code of [DAG-ERC](https://github.com/shenwzh3/DAG-ERC), which is quite clear.

## main dependency libs

This framework mainly includes some libs below:

- Pytorch 1.14
- Python 3.9
- transformers 4.23.1
- scikit-learn

## Customized functionalities

- Linear learning rate scheduler
- separate learning rate training(separate params group)
- flexible metric selection provided by sklearn
- detailed log provided by logging library

## Example

The framework uses finetune bert-base-uncased on **SST-2**(downloaded by hugging face datasets) as a demo

The best output is as below:

```
[2022-12-16 18:15:02,690][run.py][line:194][INFO] Epoch: 2, train_loss: 0.1376, train_acc: 95.1, train_fscore: 95.1, valid_loss: 0.2641, valid_acc: 92.43, valid_fscore: 92.43, test_loss: nan, test_acc: nan, test_fscore: 0.0, time: 131.87 sec
```

