import dspy


def db_metric(example: dspy.Example, pred: dspy.Prediction, target: str = None):
    return pred.eval_report['success']
