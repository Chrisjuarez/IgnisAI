"""The FIRMS history deploy tool creates nothing outside the Ignis account."""
import pytest

from tools import deploy_firms_history_lambda as deploy

IGNIS, OTHER = "111122223333", "444455556666"


class FakeSession:
    def __init__(self, account):
        self.account = account

    def client(self, name):
        assert name == "sts"
        return type("Sts", (), {"get_caller_identity": lambda _self: {"Account": self.account}})()


def test_refuses_credentials_for_another_account():
    with pytest.raises(SystemExit, match="6666"):
        deploy.confirm_account(FakeSession(OTHER), IGNIS)


def test_goes_ahead_in_the_ignis_account():
    deploy.confirm_account(FakeSession(IGNIS), IGNIS)


def test_the_lambda_checks_the_account_on_its_own_writes():
    env = deploy.lambda_environment("s3://ignis-bucket/history/firms", IGNIS, "key")

    assert env["Variables"]["IGNIS_AWS_ACCOUNT_ID"] == IGNIS
    assert env["Variables"]["FIRMS_HISTORY_S3"] == "s3://ignis-bucket/history/firms"


def test_refuses_the_default_profile():
    with pytest.raises(SystemExit, match="default"):
        deploy.main(["--profile", "default", "--account", IGNIS, "--region", "us-west-1",
                     "--store", "s3://ignis-bucket/history/firms"])
