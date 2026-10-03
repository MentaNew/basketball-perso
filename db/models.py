"""SQLAlchemy models: the single source of truth for the schema.

Create the tables with `python -m db.init_db`. The loader uses these tables for its
`INSERT ... ON CONFLICT` upserts; the ML and app code read through `db.connection.query_df`.
"""
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    bref_id = Column(Text, unique=True, nullable=False)  # identity (e.g. "jamesle01")
    name = Column(Text, nullable=False)                  # display name, HOF '*' stripped
    search_name = Column(Text, nullable=False)           # lower-case, accents removed
    birth_year = Column(Integer)                         # approx: season - age - 1


# Per-game and advanced stats live in one wide row per (player, season, team).
# Traded players get one row per team plus a combined row with team_abr = 'TOT';
# `is_primary` marks the single row to use for player-level analysis.
PER_GAME_COLS = [
    "g", "gs", "mp", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct", "ft", "fta",
    "ft_pct", "efg_pct", "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf", "pts",
]
ADVANCED_COLS = [
    "mp_total", "per", "ts_pct", "fg3a_rate", "fta_rate", "orb_pct", "drb_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct", "ows", "dws", "ws", "ws_per_48",
    "obpm", "dbpm", "bpm", "vorp",
]


class PlayerSeason(Base):
    __tablename__ = "player_seasons"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "team_abr"),
        Index("idx_ps_season", "season"),
        Index("idx_ps_player", "player_id"),
    )
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(Integer, nullable=False)       # ending year: 2024 = 2023-24
    team_abr = Column(Text, nullable=False)        # 'TOT' for the combined row
    team_list = Column(Text)                       # "DAL,LAL" for TOT rows
    is_primary = Column(Boolean, nullable=False, default=True)
    age = Column(Integer)
    pos = Column(Text)


# Declarative classes accept columns added after definition.
for _col in PER_GAME_COLS + ADVANCED_COLS:
    setattr(PlayerSeason, _col, Column(_col, Float))


class TeamSeason(Base):
    __tablename__ = "team_seasons"
    __table_args__ = (UniqueConstraint("team_abr", "season"), Index("idx_ts_season", "season"))
    id = Column(Integer, primary_key=True)
    team_abr = Column(Text, nullable=False)        # BRef abbreviation of that season
    season = Column(Integer, nullable=False)
    franchise_id = Column(Text, nullable=False)    # current abbreviation (SEA -> OKC)
    full_name = Column(Text)
    conference = Column(Text)                      # 'E' | 'W'
    division = Column(Text)
    w = Column(Integer)
    l = Column(Integer)  # noqa: E741
    w_pct = Column(Float)
    gb = Column(Float)
    ps_g = Column(Float)
    pa_g = Column(Float)
    srs = Column(Float)
    made_playoffs = Column(Boolean)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (Index("idx_games_season", "season"), Index("idx_games_date", "game_date"))
    game_id = Column(Text, primary_key=True)       # BRef box score id, e.g. 202310240DEN
    season = Column(Integer, nullable=False)
    game_date = Column(Date, nullable=False)
    away_team = Column(Text, nullable=False)       # abbreviations
    home_team = Column(Text, nullable=False)
    away_name = Column(Text)
    home_name = Column(Text)
    away_score = Column(Integer)
    home_score = Column(Integer)
    home_win = Column(Boolean)
    overtimes = Column(Text)
    attendance = Column(Integer)
    remarks = Column(Text)
    game_type = Column(Text)  # 'regular' | 'play_in' | 'playoff' | 'cup_final'


class EtlCheckpoint(Base):
    __tablename__ = "etl_checkpoint"
    season = Column(Integer, primary_key=True)
    step = Column(Text, primary_key=True)
    rows = Column(Integer)
    completed_at = Column(DateTime, server_default=func.now())
