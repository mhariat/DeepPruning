import json
import matplotlib.pyplot as plt
root = '/home/marwane/PycharmProjects/experiments/stats'
methods = ['Improved-Eigen-Damage/tucker_correct_back', 'Eigen-Damage', 'L1', 'OBS', 'OBD', 'Taylor']
indicators = ['Indicator/total_flops', 'Performance/val_acc', 'Indicator/prune_ratio']

results = {}
for method in methods:
    results[method] = {}
    if '/' in method:
        first, second = method.split('/')
        to_open = '{}/{}/{}/stats_run_0.json'.format(root, first, second)
    else:
        to_open = '{}/{}/stats_run_0.json'.format(root, method)
    with open(to_open) as json_file:
        data = json.load(json_file)
        for k in data.keys():
            if int(k) % 2 == 0:
                if data[k]['Indicator/prune_ratio'] < 0.99:
                    for indicator in indicators:
                        if indicator not in results[method].keys():
                            results[method][indicator] = []
                        results[method][indicator].append(data[k][indicator])

color = {'Eigen-Damage': 'red',
         'Improved-Eigen-Damage/tucker_correct_back': 'blue',
         'L1': 'purple',
         'OBS': 'magenta',
         'OBD': 'cyan',
         'Taylor': 'green'}

# color = {'Improved-Eigen-Damage/tucker': 'blue',
#          'Improved-Eigen-Damage/tucker_correct': 'red',
#          'Improved-Eigen-Damage/tucker_correct_back': 'purple',
#          'Improved-Eigen-Damage/tucker_correct_back_nosua': 'magenta',
#          'Improved-Eigen-Damage/tucker_correct_back_distillation': 'green',
#          'Eigen-Damage': 'grey'}

for method in methods:
    X = results[method]['Indicator/prune_ratio']
    Y = results[method]['Indicator/total_flops']
    plt.plot(X, Y, color[method], label=method, linewidth=3)

plt.legend(methods)
plt.xlabel('Pruning Ratio')
plt.ylabel('Flops')
plt.title('Evolution of flops with compression')

conv1 = []
conv5 = []
conv10 = []
conv15 = []
conv20 = []
conv25 = []
conv30 = []
linear = []

l = results['Improved-Eigen-Damage/tucker_correct_back']['Indicator/prune_ratio_per_module']
l = results['Eigen-Damage']['Indicator/prune_ratio_per_module']
for e in l:
    conv1.append(e['Conv_1'])
    conv5.append(e['Conv_5'])
    conv10.append(e['Conv_10'])
    conv15.append(e['Conv_15'])
    conv20.append(e['Conv_20'])
    conv25.append(e['Conv_25'])
    conv30.append(e['Conv_30'])
    linear.append(e['Linear_33'])

X = results[method]['Indicator/prune_ratio']
plt.plot(X, conv1, linewidth=3)
plt.plot(X, conv5, linewidth=3)
plt.plot(X, conv10, linewidth=3)
plt.plot(X, conv15, linewidth=3)
plt.plot(X, conv20, linewidth=3)
plt.plot(X, conv25, linewidth=3)
plt.plot(X, conv30, linewidth=3)
plt.plot(X, linear, linewidth=3)

plt.legend(['Conv1', 'Conv5', 'Conv10', 'Conv15', 'Conv20', 'Conv25', 'Conv30', 'Linear'])
plt.xlabel('Pruning Ratio')
plt.ylabel('Compression')
plt.title('Evolution of compression with conv/fc layers')