import kfp
from kfp import dsl
import requests

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='dtro/preprocess-model:0.1',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'x_valid': '/app/x_valid.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
            'y_valid': '/app/y_valid.npy',
            'retrain': '/app/retrain.npy'
        }
    )

def train_op(x_train, y_train,x_valid,y_valid):

    return dsl.ContainerOp(
        name='Train Model',
        image='dtro/train-model:0.1',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train,
            '--x_valid', x_valid,
            '--y_valid', y_valid
        ],
        file_outputs={
            'model': '/app/model.h5'
        }
    )

def re_train_op(x_train, y_train,x_valid,y_valid, model):

    return dsl.ContainerOp(
        name='Retrain Model',
        image='dtro/retrain-model:0.1',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train,
            '--x_valid', x_valid,
            '--y_valid', y_valid,
            '--model' , model
        ],
        file_outputs={
            'model': '/app/model.h5'
        }
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='dtro/test-model:0.1',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt'
        }
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='dtro/deploy-model:0.1',
        arguments=[
            '--model', model
        ]
    )

def check_model_is_exists():
  ## url model 
  URL = "https://github.com/dtroo/KLTN/raw/main/Model/model.h5"
  r = requests.get(URL)
  if r.status_code == 200:
    return True
  return False

@dsl.pipeline(
   name='Machine learning Pipeline',
   description='Create or retrain model pipeline'
)
def COTM_pipeline():
    _preprocess_op = preprocess_op()
    if(check_model_is_exists() == True):
        _train_op = re_train_op(
            dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_train']),
            dsl.InputArgumentPath(_preprocess_op.outputs['x_valid']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_valid']),
            dsl.InputArgumentPath(_preprocess_op.outputs['model'])
        ).after(_preprocess_op)

        _test_op = test_op(
            dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
            dsl.InputArgumentPath(_train_op.outputs['model'])
        ).after(_train_op)

        deploy_model_op(
            dsl.InputArgumentPath(_train_op.outputs['model'])
        ).after(_test_op)
    
    else:
        _train_op = train_op(
            dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_train']),
            dsl.InputArgumentPath(_preprocess_op.outputs['x_valid']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_valid'])
        ).after(_preprocess_op)

        _test_op = test_op(
            dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
            dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
            dsl.InputArgumentPath(_train_op.outputs['model'])
        ).after(_train_op)

        deploy_model_op(
            dsl.InputArgumentPath(_train_op.outputs['model'])
        ).after(_test_op)

client = kfp.Client()
client.create_run_from_pipeline_func(COTM_pipeline, arguments={})