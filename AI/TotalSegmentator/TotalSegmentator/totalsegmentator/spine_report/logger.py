from totalsegmentator.reporting import setup_logger as _setup_logger


def setup_logger(log_file):
    return _setup_logger(log_file, name=f"totalseg_spine_report.{log_file}")
