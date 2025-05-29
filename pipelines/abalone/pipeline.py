"""Movie Recommendation System Pipeline

Pipeline workflow:
    Process (Movies + Credits) -> Train -> Evaluate -> Condition -> ModelStep

Implements a get_pipeline(**kwargs) method for movie recommendation system.
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
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session

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

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="MovieRecommendationPackageGroup",
    pipeline_name="MovieRecommendationPipeline",
    base_job_prefix="MovieRecommendation",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance for movie recommendation system.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
        model_package_group_name: Name of the model package group
        pipeline_name: Name of the pipeline
        base_job_prefix: Prefix for job names

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    # Input data parameters - separate for movies and credits files
    movies_input_data = ParameterString(
        name="MoviesInputDataUrl",
        default_value=f"s3://{default_bucket}/movie-recommendation-data/tmdb_5000_movies.csv",
    )
    credits_input_data = ParameterString(
        name="CreditsInputDataUrl", 
        default_value=f"s3://{default_bucket}/movie-recommendation-data/tmdb_5000_credits.csv",
    )

    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-movie-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=movies_input_data,
                destination="/opt/ml/processing/input/movies",
                input_name="movies"
            ),
            ProcessingInput(
                source=credits_input_data,
                destination="/opt/ml/processing/input/credits", 
                input_name="credits"
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=[
            "--movies-data", "/opt/ml/processing/input/movies/tmdb_5000_movies.csv",
            "--credits-data", "/opt/ml/processing/input/credits/tmdb_5000_credits.csv"
        ],
    )
    step_process = ProcessingStep(
        name="PreprocessMovieData",
        step_args=step_args,
    )

    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/MovieRecommendationTrain"
    
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir=BASE_DIR,
        framework_version="1.2-1",
        py_version="py3",
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/movie-recommendation-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = sklearn_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    step_train = TrainingStep(
        name="TrainMovieRecommendationModel",
        step_args=step_args,
    )

    # Processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type=processing_instance_type,
        ),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-movie-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="MovieRecommendationEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateMovieRecommendationModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
    # Create sklearn model for registration
    sklearn_model = Model(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type="ml.m5.large",
        ),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
        entry_point="inference.py",
        source_dir=BASE_DIR,
    )
    
    step_args = sklearn_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="RegisterMovieRecommendationModel",
        step_args=step_args,
    )

    # Condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="recommendation_metrics.quality_score"
        ),
        right=0.3,  # Quality threshold for model approval
    )
    step_cond = ConditionStep(
        name="CheckMovieRecommendationQuality",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            training_instance_count,
            model_approval_status,
            movies_input_data,
            credits_input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
