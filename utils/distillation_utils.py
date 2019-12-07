import torch
import gc
import torch.nn as nn
import torch.nn.functional as F


def loss_kd(output_student, output_teacher, target, alpha, temperature):
    log_student_softmax = F.log_softmax(output_student/temperature, dim=1)
    teacher_softmax = F.softmax(output_teacher/temperature, dim=1)
    loss_1 = nn.KLDivLoss()
    loss_2 = nn.CrossEntropyLoss()
    kd_loss = loss_1(log_student_softmax, teacher_softmax)*(temperature**2)*alpha +\
              loss_2(output_student, target)*(1 - alpha)
    return kd_loss


def train_kd(student, teacher, optimizer, use_cuda, train_dataloader, epoch, alpha, temperature, log_interval=100):
    student.train()
    iteration = 0
    train_loss = 0
    correct = 0
    nb_iterations = int(len(train_dataloader.dataset)/train_dataloader.batch_size)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            teacher.cuda()
        optimizer.zero_grad()
        output_student = student(data)
        with torch.no_grad():
            output_teacher = teacher(data)
        loss = loss_kd(output_student, output_teacher, target, alpha, temperature)
        loss.backward()
        optimizer.step()
        pred = output_student.data.max(1, keepdim=True)[1]
        correct_batch = pred.eq(target.data.view_as(pred)).sum().item()
        correct += correct_batch
        train_loss = loss.data.item()
        train_accuracy = correct_batch/train_dataloader.batch_size
        if iteration % log_interval == 0:
            print('\nTrain Epoch: {}. Iteration: [{:.0f}/{:.0f}]. Loss: {:.6f}. Accuracy: {:.2f}%.\n'.
                  format(epoch, iteration, nb_iterations, train_loss, 100 * train_accuracy))
        iteration += 1
    train_accuracy = correct/len(train_dataloader.dataset)
    train_loss /= len(train_dataloader.dataset)
    torch.cuda.empty_cache()
    gc.collect()
    teacher.cpu()
    return train_accuracy, train_loss
