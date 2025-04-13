import argparse


class Task():
    def __init__(self, mode):
        self.mode = mode

    def run(self):
        pass

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--output_file_path', type=str, default='results/eval.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    task = Task(args.mode)
    task.run()