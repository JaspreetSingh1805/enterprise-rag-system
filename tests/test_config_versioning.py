import pytest
from pydantic import ValidationError

from app.core.exceptions import ConfigurationError
from app.models.domain import SearchMode
from app.models.pipeline_config import (
    PipelineConfig,
    RetrievalConfig,
)
from app.pipeline.config_manager import DynamicConfigManager


def test_config_version_validation():
    # Valid versions
    c1 = PipelineConfig(version="v1")
    assert c1.version == "v1"
    c2 = PipelineConfig(version="v2.1")
    assert c2.version == "v2.1"

    # Invalid versions
    with pytest.raises(ValidationError):
        PipelineConfig(version="invalid_version_1")

    with pytest.raises(ValidationError):
        PipelineConfig(version="version-2")


def test_dynamic_config_activation(metadata_store, in_memory_vector_store):
    mgr = DynamicConfigManager(metadata_store, in_memory_vector_store)

    # Initial default is v1
    active = mgr.get_active_config()
    assert active.version == "v1"
    assert active.is_active is True

    # Create new config v2
    v2 = PipelineConfig(
        version="v2",
        description="Version 2 with hybrid search and reranking",
        retrieval=RetrievalConfig(search_mode=SearchMode.HYBRID, top_k=8),
    )
    mgr.save_new_config(v2)

    # List configs
    configs = mgr.list_configs()
    assert len(configs) >= 2

    # Activate v2 without restarting
    activated = mgr.activate_config("v2")
    assert activated.version == "v2"
    assert activated.is_active is True

    # Verify that get_active_config now returns v2
    current_active = mgr.get_active_config()
    assert current_active.version == "v2"
    assert current_active.retrieval.top_k == 8

    # Attempting to activate non-existent version raises ConfigurationError
    with pytest.raises(ConfigurationError):
        mgr.activate_config("v999")
