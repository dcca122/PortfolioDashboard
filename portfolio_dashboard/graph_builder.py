"""Graph construction utilities for PortfolioDashboard."""
from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd
import networkx as nx


def build_sector_graph(sectors: Dict[str, str]) -> nx.Graph:
    """Create a graph connecting stocks that share the same sector."""
    g = nx.Graph()
    for ticker, sector in sectors.items():
        g.add_node(ticker, sector=sector)
    # connect nodes within the same sector
    sector_groups: Dict[str, list[str]] = {}
    for ticker, sector in sectors.items():
        sector_groups.setdefault(sector, []).append(ticker)
    for tickers in sector_groups.values():
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1 :]:
                g.add_edge(t1, t2, weight=1.0)
    return g


def build_correlation_graph(returns: pd.DataFrame, threshold: float = 0.8) -> nx.Graph:
    """Create a graph using pairwise correlation of returns."""
    corr = returns.unstack("Ticker").corr()
    g = nx.Graph()
    for ticker in corr.index:
        g.add_node(ticker)
    for t1 in corr.index:
        for t2 in corr.columns:
            if t1 >= t2:
                continue
            if corr.loc[t1, t2] >= threshold:
                g.add_edge(t1, t2, weight=float(corr.loc[t1, t2]))
    return g


def compute_graph_features(g: nx.Graph) -> pd.DataFrame:
    """Compute basic graph metrics for each node."""
    degree = dict(g.degree())
    pagerank = nx.pagerank(g) if g.number_of_edges() > 0 else {n: 0.0 for n in g.nodes}
    df = pd.DataFrame({
        'degree': pd.Series(degree),
        'pagerank': pd.Series(pagerank)
    })
    return df

