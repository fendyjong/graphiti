"""Unit tests for `EpisodicNode.attributes` / `EpisodicEdge.attributes`.

`EntityNode` and `EntityEdge` have always carried an open-ended `attributes`
map that is persisted as first-class graph properties. Episodic nodes and
their MENTIONS edges did not, so callers had no supported way to put their own
properties on an episode -- and any property they set out-of-band was erased by
the next save, because the episode save wrote `SET n = {...fixed key set...}`.

These tests lock in both halves of the fix:
  * attributes are written on every provider, single-save and bulk;
  * the save merges (`SET n +=`) instead of replacing (`SET n =`), so
    properties graphiti did not write survive a re-save by uuid.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EpisodicEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.models.edges.edge_db_queries import (
    get_episodic_edge_save_bulk_query,
    get_episodic_edge_save_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges

NON_KUZU_PROVIDERS = [p for p in GraphProvider if p != GraphProvider.KUZU]


def _episode(**overrides) -> EpisodicNode:
    kwargs = {
        'name': 'ep-1',
        'group_id': 'group',
        'source': EpisodeType.text,
        'source_description': 'test',
        'content': 'hello',
        'valid_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
        'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return EpisodicNode(**kwargs)


def _episodic_edge(**overrides) -> EpisodicEdge:
    kwargs = {
        'group_id': 'group',
        'source_node_uuid': 'episode-uuid',
        'target_node_uuid': 'entity-uuid',
        'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return EpisodicEdge(**kwargs)


# --------------------------------------------------------------------------- #
# model surface
# --------------------------------------------------------------------------- #


def test_episodic_node_attributes_default_empty():
    assert _episode().attributes == {}


def test_episodic_node_accepts_attributes():
    node = _episode(attributes={'tenant_id': 'org_a', 'document_id': 'doc_42'})
    assert node.attributes == {'tenant_id': 'org_a', 'document_id': 'doc_42'}


def test_episodic_node_attributes_are_assignable():
    # `Node` sets validate_assignment=True, so an undeclared attribute cannot be
    # assigned after construction. A declared field can.
    node = _episode()
    node.attributes = {'tenant_id': 'org_a'}
    assert node.attributes == {'tenant_id': 'org_a'}


def test_episodic_edge_attributes_default_empty():
    assert _episodic_edge().attributes == {}


def test_episodic_edge_accepts_attributes():
    edge = _episodic_edge(attributes={'tenant_id': 'org_a'})
    assert edge.attributes == {'tenant_id': 'org_a'}


def test_episodic_edge_attributes_are_assignable():
    edge = _episodic_edge()
    edge.attributes = {'tenant_id': 'org_a'}
    assert edge.attributes == {'tenant_id': 'org_a'}


# --------------------------------------------------------------------------- #
# query shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episode_save_query_merges_attributes(provider):
    query = get_episode_node_save_query(provider)
    assert 'SET n += $attributes' in query


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episode_bulk_save_query_merges_attributes(provider):
    query = get_episode_node_save_bulk_query(provider)
    assert 'SET n += episode.attributes' in query


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episode_save_queries_do_not_replace_the_property_map(provider):
    # `SET n = {...}` deletes every property not in the literal, which would
    # erase caller attributes on any re-save by uuid (and on a save built from
    # an episode read back from the graph, since reads do not repopulate
    # `attributes`). The declared fields must be merged, not assigned.
    for query in (
        get_episode_node_save_query(provider),
        get_episode_node_save_bulk_query(provider),
    ):
        assert 'SET n = {' not in query
        assert 'SET n += {' in query


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episode_save_queries_apply_attributes_before_declared_fields(provider):
    # Declared episode fields must win over a same-named attribute, mirroring
    # the `if k not in entity_data` guard on the entity paths.
    for query, attributes_clause in (
        (get_episode_node_save_query(provider), 'SET n += $attributes'),
        (get_episode_node_save_bulk_query(provider), 'SET n += episode.attributes'),
    ):
        assert query.index(attributes_clause) < query.index('SET n += {')


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episodic_edge_save_queries_merge_attributes(provider):
    assert 'SET e += $attributes' in get_episodic_edge_save_query(provider)
    assert 'SET e += edge.attributes' in get_episodic_edge_save_bulk_query(provider)


@pytest.mark.parametrize('provider', NON_KUZU_PROVIDERS)
def test_episodic_edge_save_queries_rewrite_every_declared_field_after_the_splat(provider):
    # MERGE has already created the edge by the time `SET e += <attributes>`
    # runs, so a declared property the following SET block does not rewrite can
    # be hijacked by a same-named attribute. `uuid` is the dangerous one: it is
    # only ever written by the MERGE pattern, so overwriting it leaves the edge
    # unfindable by its logical uuid, duplicated on the next save, and missed by
    # `delete_by_uuids`.
    for query, splat, param in (
        (get_episodic_edge_save_query(provider), 'SET e += $attributes', '$'),
        (get_episodic_edge_save_bulk_query(provider), 'SET e += edge.attributes', 'edge.'),
    ):
        after_splat = query[query.index(splat) + len(splat) :]
        for field in ('uuid', 'group_id', 'created_at'):
            assert f'e.{field} = {param}{field}' in after_splat


def test_kuzu_episode_save_queries_write_serialized_attributes():
    # Kuzu has an explicit schema and stores attributes as a JSON string
    # column, exactly as it already does for Entity / RelatesToNode_.
    for query in (
        get_episode_node_save_query(GraphProvider.KUZU),
        get_episode_node_save_bulk_query(GraphProvider.KUZU),
    ):
        assert 'n.attributes = $attributes' in query


def test_kuzu_episodic_edge_save_queries_write_serialized_attributes():
    for query in (
        get_episodic_edge_save_query(GraphProvider.KUZU),
        get_episodic_edge_save_bulk_query(GraphProvider.KUZU),
    ):
        assert 'e.attributes = $attributes' in query


def test_kuzu_schema_declares_episodic_attribute_columns():
    pytest.importorskip('kuzu', reason='kuzu_driver imports the kuzu package at module level')
    from graphiti_core.driver.kuzu_driver import SCHEMA_QUERIES

    episodic_table = SCHEMA_QUERIES.split('CREATE NODE TABLE IF NOT EXISTS Episodic (')[1].split(
        ');'
    )[0]
    assert 'attributes STRING' in episodic_table

    mentions_table = SCHEMA_QUERIES.split('CREATE REL TABLE IF NOT EXISTS MENTIONS(')[1].split(
        ');'
    )[0]
    assert 'attributes STRING' in mentions_table


# --------------------------------------------------------------------------- #
# write path
# --------------------------------------------------------------------------- #


class _RecordingTx:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))

    def params_for(self, marker: str) -> dict:
        matching = [kwargs for query, kwargs in self.calls if marker in query]
        assert matching, f'no query containing {marker!r} was run'
        return matching[0]


class _FakeDriver:
    graph_operations_interface = None

    def __init__(self, provider: GraphProvider):
        self.provider = provider


class _UnusedEmbedder:
    async def create(self, input_data):  # pragma: no cover - entity lists are empty
        raise AssertionError('embedder should not be called without entity nodes/edges')


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.FALKORDB, GraphProvider.NEO4J])
async def test_bulk_save_carries_episodic_attributes(provider):
    # A datetime value on purpose: for string-valued attributes the non-Kuzu
    # path is near-identity, so a test using only strings stays green even if
    # the serialization step is deleted outright. Datetimes are the one thing
    # these providers cannot store raw, so asserting the normalized form makes
    # this bite on every provider rather than only on Kuzu.
    attributes = {
        'tenant_id': 'org_a',
        'ingested_at': datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    }
    expected = {'tenant_id': 'org_a', 'ingested_at': '2024-01-02T03:04:05+00:00'}
    tx = _RecordingTx()

    await add_nodes_and_edges_bulk_tx(
        tx,  # type: ignore[arg-type]
        [_episode(attributes=attributes)],
        [_episodic_edge(attributes=attributes)],
        [],
        [],
        _UnusedEmbedder(),  # type: ignore[arg-type]
        _FakeDriver(provider),  # type: ignore[arg-type]
    )

    episodes = tx.params_for('MERGE (n:Episodic')['episodes']
    assert episodes[0]['attributes'] == expected

    episodic_edges = tx.params_for('MERGE (episode)-[e:MENTIONS')['episodic_edges']
    assert episodic_edges[0]['attributes'] == expected


@pytest.mark.asyncio
async def test_bulk_save_serializes_episodic_attributes_for_kuzu():
    attributes = {'tenant_id': 'org_a', 'document_id': 'doc_42'}
    tx = _RecordingTx()

    await add_nodes_and_edges_bulk_tx(
        tx,  # type: ignore[arg-type]
        [_episode(attributes=attributes)],
        [_episodic_edge(attributes=attributes)],
        [],
        [],
        _UnusedEmbedder(),  # type: ignore[arg-type]
        _FakeDriver(GraphProvider.KUZU),  # type: ignore[arg-type]
    )

    assert json.loads(tx.params_for('MERGE (n:Episodic')['attributes']) == attributes
    assert json.loads(tx.params_for('MERGE (episode)-[e:MENTIONS')['attributes']) == attributes


@pytest.mark.asyncio
async def test_bulk_save_without_attributes_still_sends_an_empty_map():
    # Additive: an episode with no attributes must produce the same properties
    # it always did, and `SET n += {}` must stay a no-op rather than a null.
    tx = _RecordingTx()

    await add_nodes_and_edges_bulk_tx(
        tx,  # type: ignore[arg-type]
        [_episode()],
        [_episodic_edge()],
        [],
        [],
        _UnusedEmbedder(),  # type: ignore[arg-type]
        _FakeDriver(GraphProvider.FALKORDB),  # type: ignore[arg-type]
    )

    assert tx.params_for('MERGE (n:Episodic')['episodes'][0]['attributes'] == {}
    assert tx.params_for('MERGE (episode)-[e:MENTIONS')['episodic_edges'][0]['attributes'] == {}


# --------------------------------------------------------------------------- #
# the _build_episodic_edges seam
# --------------------------------------------------------------------------- #


def _bare_graphiti(**attrs):
    """A Graphiti with no __init__ run.

    `_build_episodic_edges` reads no instance state, and `_process_episode_data`
    with `saga=None` touches only the three attributes set here -- so the real
    methods can be exercised without a live driver, LLM or embedder.
    """
    instance = Graphiti.__new__(Graphiti)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


def test_build_episodic_edges_delegates_to_the_module_function():
    # The seam exists to be overridden, so it must stay a pure pass-through:
    # same edges the module-level builder produces, nothing added or reordered.
    nodes = [EntityNode(name=f'n{i}', group_id='group') for i in range(3)]
    episodes = [_episode(), _episode()]
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    seam = _bare_graphiti()._build_episodic_edges(nodes, episodes, now)
    direct = build_episodic_edges(nodes, [ep.uuid for ep in episodes], now)

    assert [(e.source_node_uuid, e.target_node_uuid, e.group_id) for e in seam] == [
        (e.source_node_uuid, e.target_node_uuid, e.group_id) for e in direct
    ]
    assert len(seam) == len(nodes) * len(episodes)


def test_build_episodic_edges_forwards_the_node_episode_index_map():
    # Attribution: with a map, each node connects only to its own episodes.
    # Dropping the argument would silently fan every node out to every episode.
    nodes = [EntityNode(name='n0', group_id='group'), EntityNode(name='n1', group_id='group')]
    episodes = [_episode(), _episode()]
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index_map = {nodes[0].uuid: [0], nodes[1].uuid: [1]}

    edges = _bare_graphiti()._build_episodic_edges(nodes, episodes, now, index_map)

    assert len(edges) == 2
    assert {(e.source_node_uuid, e.target_node_uuid) for e in edges} == {
        (episodes[0].uuid, nodes[0].uuid),
        (episodes[1].uuid, nodes[1].uuid),
    }


@pytest.mark.asyncio
async def test_process_episode_data_saves_the_edges_the_seam_returns():
    """The seam is only useful if `_process_episode_data` routes through it.

    A subclass that tags the edges it builds must see those tags reach the bulk
    save -- that is the whole point, since MENTIONS edges are created AND saved
    inside this method and are never exposed to the caller beforehand.
    """
    tagged = {'tenant_id': 'org_a'}

    class _TaggingGraphiti(Graphiti):
        def _build_episodic_edges(self, nodes, episodes, now, node_episode_index_map=None):
            edges = super()._build_episodic_edges(nodes, episodes, now, node_episode_index_map)
            for edge in edges:
                edge.attributes = {**edge.attributes, **tagged}
            return edges

    instance = _TaggingGraphiti.__new__(_TaggingGraphiti)
    instance.driver = _FakeDriver(GraphProvider.FALKORDB)
    instance.embedder = _UnusedEmbedder()
    instance.store_raw_episode_content = True

    saved = {}

    async def _capture(driver, episodes, episodic_edges, nodes, entity_edges, embedder):
        saved['episodic_edges'] = episodic_edges

    with patch('graphiti_core.graphiti.add_nodes_and_edges_bulk', new=_capture):
        episodic_edges, _primary = await Graphiti._process_episode_data(
            instance,
            _episode(),
            [EntityNode(name='n0', group_id='group')],
            [],
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            'group',
        )

    assert len(saved['episodic_edges']) == 1
    assert saved['episodic_edges'][0].attributes == tagged
    assert saved['episodic_edges'] is episodic_edges


# --------------------------------------------------------------------------- #
# Kuzu schema migration
# --------------------------------------------------------------------------- #


def test_kuzu_setup_schema_adds_attributes_to_a_pre_existing_database(tmp_path):
    """`CREATE ... IF NOT EXISTS` is a no-op on a database that already has the
    table, so it never adds a column declared later. Without the ALTER pass an
    upgraded database keeps the old shape and every episode write dies with
    "Binder exception: Cannot find property attributes"."""
    kuzu = pytest.importorskip('kuzu')

    from graphiti_core.driver.kuzu_driver import SCHEMA_MIGRATION_QUERIES, SCHEMA_QUERIES

    db_path = str(tmp_path / 'db')
    # A database created before `attributes` existed.
    legacy = kuzu.Connection(kuzu.Database(db_path))
    legacy.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Episodic (
            uuid STRING PRIMARY KEY, name STRING, group_id STRING, created_at TIMESTAMP,
            source STRING, source_description STRING, content STRING, valid_at TIMESTAMP,
            entity_edges STRING[]
        );
        CREATE NODE TABLE IF NOT EXISTS Entity (uuid STRING PRIMARY KEY, name STRING);
        CREATE REL TABLE IF NOT EXISTS MENTIONS(
            FROM Episodic TO Entity, uuid STRING PRIMARY KEY, group_id STRING,
            created_at TIMESTAMP
        );
        """
    )
    with pytest.raises(Exception, match='attributes'):
        legacy.execute("MERGE (n:Episodic {uuid: 'a'}) SET n.attributes = 'x'")
    legacy.close()

    conn = kuzu.Connection(kuzu.Database(db_path))
    conn.execute(SCHEMA_QUERIES)
    conn.execute(SCHEMA_MIGRATION_QUERIES)
    # Idempotent: setup_schema runs on every driver construction, not just once.
    conn.execute(SCHEMA_MIGRATION_QUERIES)

    conn.execute("MERGE (n:Episodic {uuid: 'a'}) SET n.attributes = 'x'")
    conn.execute("MERGE (n:Entity {uuid: 'b'})")
    conn.execute(
        """
        MATCH (e:Episodic {uuid: 'a'}), (n:Entity {uuid: 'b'})
        MERGE (e)-[m:MENTIONS {uuid: 'm1'}]->(n)
        SET m.attributes = 'y'
        """
    )
    assert conn.execute("MATCH (n:Episodic {uuid: 'a'}) RETURN n.attributes").get_next() == ['x']
    assert conn.execute('MATCH ()-[m:MENTIONS]->() RETURN m.attributes').get_next() == ['y']
    conn.close()
