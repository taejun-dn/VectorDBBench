from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    IVFFlatTypedDictN,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from vectordb_bench.backend.clients.milvus.cli import (
    MilvusTypedDict,
    MilvusHNSWTypedDict,
)

DBTYPE = DB.Milvus


class DnMilvusVdpuHNSWTypedDict(CommonTypedDict, MilvusTypedDict, MilvusHNSWTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(DnMilvusVdpuHNSWTypedDict)
def DnMilvusVdpuHNSW(**parameters: Unpack[DnMilvusVdpuHNSWTypedDict]):
    from .config import DnMilvusVdpuHNSWConfig, MilvusConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
        ),
        db_case_config=DnMilvusVdpuHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_search"],
        ),
        **parameters,
    )