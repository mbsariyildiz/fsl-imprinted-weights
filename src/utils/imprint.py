import torch
from tqdm import tqdm

__all__ = ['imprint']

def imprint(model, novel_loader=None, n_data_augments=1, random=True, d_emb=64, device='cuda'):

    if not random:
        # switch to evaluate mode
        model.eval()

        first = True # first batch indicator
        pbar = tqdm(desc='imprinting', total=len(novel_loader.dataset)*n_data_augments)
        with torch.no_grad():
            for _ in range(n_data_augments):
                for batch_idx, (input, target) in enumerate(novel_loader):
                    input = input.to(device)

                    # compute output
                    output = model.extract(input)

                    if first:
                        output_stack = output
                        target_stack = target
                        first = False
                    else:
                        output_stack = torch.cat((output_stack, output), 0)
                        target_stack = torch.cat((target_stack, target), 0)

                    pbar.update(input.size(0))
        pbar.close()

    new_weight = torch.zeros(100, d_emb)
    for i in range(100):
        tmp = output_stack[target_stack == (i + 100)].mean(0) if not random else torch.randn(d_emb, device=device)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.to(device)))
    model.classifier.fc = torch.nn.Linear(d_emb, 200, bias=False).to(device)
    model.classifier.fc.weight.data = weight
    model.n_classes = 200
    model.classifier.n_classes = 200