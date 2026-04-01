"""pytest configuration for ScoringBench tests."""

import logging

# Configure pytest logging to show INFO and DEBUG levels during test runs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(name)s | %(message)s'
)

# Optionally reduce noise from specific libraries
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
logging.getLogger("pytabkit").setLevel(logging.WARNING)
