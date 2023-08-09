from ontolearn.metrics import F1, Accuracy, Precision, Recall

def quality(KB, solution, pos, neg):
    f1 = F1().score2; accuracy = Accuracy().score2; precision = Precision().score2; recall = Recall().score2
    instances = set(KB.individuals(solution))
    if isinstance(list(pos)[0], str):
        instances = {ind.get_iri().as_str().split("/")[-1] for ind in instances}
    tp=len(pos.intersection(instances))
    fn=len(pos.difference(instances))
    fp=len(neg.intersection(instances))
    tn=len(neg.difference(instances))
    print("Accuracy: {}%".format(100*accuracy(tp, fn, fp, tn)[-1]))
    print("Precision: {}%".format(100*precision(tp, fn, fp, tn)[-1]))
    print("Recall: {}%".format(100*recall(tp, fn, fp, tn)[-1]))
    print("F1: {}%".format(100*f1(tp, fn, fp, tn)[-1]))