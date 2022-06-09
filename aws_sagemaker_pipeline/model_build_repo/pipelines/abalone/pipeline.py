"""Workflow pipeline for garbage classification project

                                               . -RegisterModel
                                              .
    Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.  
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo

from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.functions import Join
from sagemaker.tensorflow import TensorFlow
import json




BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="GarbageClassificationPackageGroup",
    pipeline_name="GarbageClassificationPipeline",
    base_job_prefix="GarbageClassification",
):
    """Gets a SageMaker ML Pipeline instance working with GarbageClassification data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    
    print("BASE_DIR = ", BASE_DIR)

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
        
    # parameters for pipeline execution      
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    model_directory = "s3://abla-garbage-classification-model/xception-pretrained-weights/model/"
    acceptable_accuracy = ParameterFloat(name="AcceptableAccuracy",  default_value=0.90)
    training_instance_type = "ml.p3.2xlarge"    
    training_instance_count = 1
    train_input_path = "s3://abla-garbage-classification-data/train-and-validate/" 
    test_input_path = "s3://abla-garbage-classification-data/test/"
    processing_instance_count = 1
    processing_instance_type = "ml.m5.xlarge"
    image_width = 320    
    image_height = 320    
    image_channels = 3    
    
    # The values of the next three params were obtained by hyperparameter tuning run in a separate note book.
    epochs = 9
    batch_size = 94 
    learning_rate = 0.006

    print("Read the Parameters successfully!")   


    # Training step for generating model artifacts
    
    # Hyperparameters are tuned in a separate notebook, here we just use the best parameters we reached 
    # in the tuning notebook.
    hyperparameters={
        "epochs": epochs,
        "batch-size": batch_size,
        "learning-rate": learning_rate
    }

    # Define Metrics to be displayed in Cloud Watch
    metric_definitions = [
        {'Name': 'loss',                     'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'categorical_accuracy',     'Regex': 'categorical_accuracy: ([0-9\\.]+)'},
        {'Name': 'val_loss',                 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'val_categorical_accuracy', 'Regex': 'val_categorical_accuracy: ([0-9\\.]+)'}
    ]

    print("Defined hyperparameters and metric definitions successfully!!")   
    
    estimator = TensorFlow(
      entry_point = os.path.join(BASE_DIR, "training-script.py"),    # entry script, script it uses for training
      role = role,
      framework_version = "2.3.2", # TensorFlow's version
      hyperparameters = hyperparameters,
      instance_count = training_instance_count, # The number of GPUs instances to use
      instance_type = training_instance_type,
      metric_definitions = metric_definitions,
      py_version = 'py37',
      model_dir = model_directory,
    )

    print("Training ...")

    # Start the training
    estimator.fit()

    print("Training successful!")
    
    image_uri = estimator.training_image_uri()  
    
    step_train = TrainingStep(
        name="TrainGarbageClassificationModel",
        estimator=estimator,
    )
    
    # processing step for evaluation       
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/GarbageClassificationTrain"
    print("model_path = ", model_path)
    
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-garbage-classification-eval",
        sagemaker_session=sagemaker_session,
        role=role,
        env = {"image_width": str(image_width), "image_height": str(image_height), "image_channels": str(image_channels)},
    )
    evaluation_report = PropertyFile(
        name="GarbageClassificationEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateGarbageClassificationModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=test_input_path,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterGarbageClassificationModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg"],
        response_types=["text/plain"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],  
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=acceptable_accuracy,
    )
    step_cond = ConditionStep(
        name="CheckMacroAvgGarbageClassificationEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[      
            model_approval_status,
            acceptable_accuracy,
        ],

        steps=[step_train, step_eval, step_cond],
        
        sagemaker_session=sagemaker_session,
    )
    
    print("pipeline.py ended")    
    
    return pipeline












    #training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.p3.2xlarge")   
    
    #train_input_path = ParameterString(
    #    name="TrainInputPath", default_value=f"s3://abla-garbage-classification-data-reduced/train-and-validate/")    
    
    #test_input_path = ParameterString(
    #    name="TestInputPath", default_value=f"s3://abla-garbage-classification-data-reduced/test/")
    
    #processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    #TODO change to ml.p3.2xlarge
    #processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")     
    #image_width              = ParameterInteger(name="ImageWidth",    default_value=320)
    #image_height             = ParameterInteger(name="ImageHeight",   default_value=320)
    #image_channels           = ParameterInteger(name="ImageChannels", default_value=3)
    #epochs                   = ParameterInteger(name="Epochs",        default_value=1) ## TODO change bacxk to 20
    #batch_size               = ParameterInteger(name="BatchSize",     default_value=63)#TODO change back to 64 or other
    #learning_rate            = ParameterFloat(name="LearningRate",  default_value=0.001)