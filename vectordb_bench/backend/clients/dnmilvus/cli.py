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

DBTYPE = DB.Milvus


class MilvusTypedDict(TypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    user_name: Annotated[
        str | None,
        click.option("--user-name", type=str, help="Db username", required=False),
    ]
    password: Annotated[
        str | None,
        click.option("--password", type=str, help="Db password", required=False),
    ]
    num_shards: Annotated[
        int,
        click.option(
            "--num-shards",
            type=int,
            help="Number of shards",
            required=False,
            default=1,
            show_default=True,
        ),
    ]


class MilvusHNSWTypedDict(CommonTypedDict, MilvusTypedDict, HNSWFlavor3): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusHNSWTypedDict)
def MilvusHNSW(**parameters: Unpack[MilvusHNSWTypedDict]):
    from .config import HNSWConfig, MilvusConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
        ),
        db_case_config=HNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_search"],
        ),
        **parameters,
    )


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
            vdpu_sw_reset
        ),
        **parameters,
    )