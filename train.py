def calculate_acc(output, label):
    predicts = output.max(1, keepdim=True)[1]
    correct = predicts.eq(label.view_as(predicts)).sum()
    acc = correct.float() / predicts.shape[0]
    return acc


def train(model, device, train_iterator, optimizer_model,optimizer_centloss,
          criterion1, criterion2, weight_centloss):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (img, label) in train_iterator:
        img = img.to(device)
        label = label.to(device)
        feature, output = model(img)
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss = criterion1(output, label) + criterion2(feature, label)*weight_centloss
        loss.backward()
        optimizer_model.step()
        for param in criterion2.parameters():
            param.grad.data *= (1. / weight_centloss)
        optimizer_centloss.step()
        accuracy = calculate_acc(output, label)
        epoch_loss += loss
        epoch_acc += accuracy
        print('|Batch loss:', loss.item(), '|Batch accuracy:', accuracy.item())
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)


def eval(model, device, test_iterator, criterion1, criterion2,weight_centloss=None):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    for (img, label) in test_iterator:
        img = img.to(device)
        label = label.to(device)
        feature, output = model(img)
        loss = criterion1(output, label) + criterion2(feature, label)*weight_centloss
        accuracy = calculate_acc(output, label)
        epoch_loss += loss
        epoch_acc += accuracy
    return epoch_loss / len(test_iterator), epoch_acc / len(test_iterator)
