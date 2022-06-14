import argparse
import push

def deploy_model(model_path):
    push.push_model_to_github(model_path)
    print(f'deploying model {model_path}...')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    deploy_model(args.model)