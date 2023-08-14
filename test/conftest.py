import wandb


def pytest_sessionstart(session):
    wandb.init(mode="disabled")
