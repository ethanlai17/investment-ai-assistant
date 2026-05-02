import argparse
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import Config


def setup_logging(config: Config) -> None:
    Path(config.logs_dir).mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level, colorize=True)
    logger.add(
        f"{config.logs_dir}/investment_assistant_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="30 days",
        level=config.log_level,
    )


def run_now(config: Config) -> None:
    from orchestrator import Orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.run_pipeline()


def run_scheduled(config: Config) -> None:
    from orchestrator import Orchestrator
    orchestrator = Orchestrator(config)
    scheduler = BlockingScheduler(timezone=config.schedule_timezone)
    scheduler.add_job(
        orchestrator.run_pipeline,
        CronTrigger(
            hour=config.schedule_hour,
            minute=config.schedule_minute,
            timezone=config.schedule_timezone,
        ),
    )
    logger.info(
        f"Scheduler started — daily at {config.schedule_hour:02d}:{config.schedule_minute:02d} {config.schedule_timezone}"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Investment Assistant")
    parser.add_argument("--run-now", action="store_true", help="Run pipeline immediately")
    args = parser.parse_args()

    config = Config.load()
    setup_logging(config)
    Path(config.outputs_dir).mkdir(exist_ok=True)

    if args.run_now:
        run_now(config)
    else:
        run_scheduled(config)
