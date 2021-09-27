
def make_weights_for_balanced_classes(classes, num_classes):
    count = [0] * num_classes
    for c in classes:
        count[c] += 1
    weight_per_class = [0.] * num_classes

    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(classes)

    for idx, c in enumerate(classes):
        weight[idx] = weight_per_class[c]
    return weight
