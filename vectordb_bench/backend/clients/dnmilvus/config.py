from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType

from vectordb_bench.backend.clients.milvus.config import MilvusConfig, MilvusIndexConfig, HNSWConfig

class DnMilvusConfig(MilvusConfig): ...

class DnMilvusIndexConfig(MilvusIndexConfig): ...
        
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
        
_dnmilvus_case_config = {
    IndexType.VDPU_HNSW: VdpuHNSWConfig,
}
