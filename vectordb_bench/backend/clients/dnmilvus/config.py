from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType


class DnMilvusConfig(DBConfig):
    uri: SecretStr = "http://localhost:19530"
    user: str | None = None
    password: SecretStr | None = None
    num_shards: int = 1

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user if self.user else None,
            "password": self.password.get_secret_value() if self.password else None,
            "num_shards": self.num_shards,
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if (
            field.name in cls.common_short_configs()
            or field.name in cls.common_long_configs()
            or field.name in ["user", "password"]
        ):
            return v
        if isinstance(v, str | SecretStr) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class MilvusIndexConfig(BaseModel):
    """Base config for milvus"""

    index: IndexType
    metric_type: MetricType | None = None
    use_partition_key: bool = True  # for label-filter

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        return self.metric_type.value

class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }
        
class VdpuHNSWConfig(HNSWConfig):
    index: IndexType = IndexType.VDPU_HNSW
    
    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "M": self.M,
                "efConstruction": self.efConstruction,
            },
        }
        
    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }
        
_milvus_case_config = {
    IndexType.VDPU_HNSW: VdpuHNSWConfig,
}
