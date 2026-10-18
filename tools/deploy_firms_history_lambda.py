"""Create or update the daily FIRMS history job: an AWS Lambda on a schedule.

Each morning it keeps yesterday's near-real-time detections for the western US
in the history store, before FIRMS replaces them with the reprocessed archive.
It runs services/tilesvc/firms_history.py as is.

Safe to re-run: every step creates what is missing and updates what exists.
Nothing is created unless the profile's credentials belong to --account, the
Ignis AWS account; the Lambda checks the same account before each write.

    python tools/deploy_firms_history_lambda.py --profile <ignis admin profile> \\
        --account <Ignis account id> --region us-west-1 \\
        --store s3://ignisai-static-chrisjuarez-2026/history/firms

The NASA key is read from backend/.env and never printed.
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "services/tilesvc/firms_history.py"
NAME = "ignis-firms-history-daily"
SCHEDULE = "cron(0 8 * * ? *)"  # 08:00 UTC: yesterday's last passes have arrived


def nasa_key() -> str:
    for line in (REPO / "backend/.env").read_text().splitlines():
        if line.startswith("NASA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("NASA_API_KEY not found in backend/.env")


def package() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SOURCE, "firms_history.py")
    return buffer.getvalue()


def confirm_account(session, expected: str) -> None:
    """Stop before creating anything unless these credentials are the Ignis account's."""
    actual = session.client("sts").get_caller_identity()["Account"]
    if actual != expected:
        sys.exit(f"credentials are for AWS account …{actual[-4:]}, not the Ignis account …{expected[-4:]}; nothing was created")


def lambda_environment(store: str, account: str, key: str) -> dict:
    return {"Variables": {"FIRMS_HISTORY_S3": store, "IGNIS_AWS_ACCOUNT_ID": account, "NASA_API_KEY": key}}


def ensure_role(iam, bucket: str, prefix: str) -> str:
    from botocore.exceptions import ClientError

    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=NAME)["Role"]["Arn"]
    except ClientError:
        arn = iam.create_role(RoleName=NAME, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="Writes the daily FIRMS history files")["Role"]["Arn"]
        time.sleep(10)  # a new role takes a moment before Lambda can assume it
    iam.attach_role_policy(RoleName=NAME, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    # Only the history prefix: read and write its files, and list it so a missing
    # file reads as missing rather than forbidden.
    iam.put_role_policy(RoleName=NAME, PolicyName="firms-history-store", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*"},
            {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
             "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*"]}}},
        ],
    }))
    return arn


def ensure_function(lam, role_arn: str, store: str, account: str) -> str:
    env = lambda_environment(store, account, nasa_key())
    code = package()
    try:
        lam.get_function(FunctionName=NAME)
        lam.update_function_code(FunctionName=NAME, ZipFile=code)
        lam.get_waiter("function_updated").wait(FunctionName=NAME)
        lam.update_function_configuration(FunctionName=NAME, Role=role_arn, Environment=env, Timeout=180, MemorySize=256)
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(FunctionName=NAME, Runtime="python3.12", Role=role_arn, Handler="firms_history.lambda_handler",
                            Code={"ZipFile": code}, Timeout=180, MemorySize=256, Environment=env,
                            Description="Keeps yesterday's near-real-time FIRMS detections for the western US")
    lam.get_waiter("function_active").wait(FunctionName=NAME)
    return lam.get_function(FunctionName=NAME)["Configuration"]["FunctionArn"]


def ensure_schedule(events, lam, function_arn: str) -> None:
    rule_arn = events.put_rule(Name=NAME, ScheduleExpression=SCHEDULE, State="ENABLED",
                               Description="Daily FIRMS history")["RuleArn"]
    events.put_targets(Rule=NAME, Targets=[{"Id": "firms-history", "Arn": function_arn}])
    try:
        lam.add_permission(FunctionName=NAME, StatementId="daily-schedule", Action="lambda:InvokeFunction",
                           Principal="events.amazonaws.com", SourceArn=rule_arn)
    except lam.exceptions.ResourceConflictException:
        pass  # already allowed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--account", required=True, help="the Ignis AWS account id; nothing is created in any other")
    parser.add_argument("--store", required=True, help="s3://bucket/prefix of the history files")
    parser.add_argument("--region", required=True)
    args = parser.parse_args(argv)
    if args.profile == "default":
        sys.exit("refusing the default AWS profile: on this machine it is another job's account")
    parsed = urlparse(args.store)
    bucket, prefix = parsed.netloc, parsed.path.strip("/")

    import boto3

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    confirm_account(session, args.account)
    role_arn = ensure_role(session.client("iam"), bucket, prefix)
    function_arn = ensure_function(session.client("lambda"), role_arn, args.store, args.account)
    ensure_schedule(session.client("events"), session.client("lambda"), function_arn)
    print(f"{NAME}: deployed, runs {SCHEDULE} (UTC), writes {args.store}")


if __name__ == "__main__":
    main()
