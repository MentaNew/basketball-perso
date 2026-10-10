"""Database schema (SQLAlchemy). The single source of truth for every table.

Create it with `python -m db.init_db`. Conventions used everywhere:
- `season` is the ending year: 2024 means the 2023-24 season.
- Teams are Basketball-Reference abbreviations *of that season* (SEA in 2005, OKC in 2015);
  `franchise_id` links them across relocations.
- Players are identified by their Basketball-Reference id (`bref_id`), never by name.
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
    bref_id = Column(Text, unique=True, nullable=False)  # e.g. "jamesle01"
    name = Column(Text, nullable=False)                  # display name, Hall-of-Fame '*' stripped
    search_name = Column(Text, nullable=False)           # lower-case, accents removed
    birth_year = Column(Integer)                         # approximate: season - age - 1


class PlayerSeason(Base):
    """One wide row per (player, season, team): per-game stats + advanced stats.

    A traded player has one row per team plus a combined row with team_abr = 'TOT'.
    `is_primary` marks the single row per player-season to use for analysis: the TOT row
    if there is one, otherwise the only row.
    """
    __tablename__ = "player_seasons"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "team_abr"),
        Index("idx_ps_season", "season"),
        Index("idx_ps_player", "player_id"),
    )

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(Integer, nullable=False)
    team_abr = Column(Text, nullable=False)   # 'TOT' for the combined row
    team_list = Column(Text)                  # "DAL,LAL" on a TOT row, else the team itself
    is_primary = Column(Boolean, nullable=False)
    age = Column(Integer)
    pos = Column(Text)

    # Per-game stats (NBA_{season}_per_game.html)
    g = Column(Float)          # games played
    gs = Column(Float)         # games started
    mp = Column(Float)         # minutes per game
    fg = Column(Float)
    fga = Column(Float)
    fg_pct = Column(Float)
    fg3 = Column(Float)
    fg3a = Column(Float)
    fg3_pct = Column(Float)
    ft = Column(Float)
    fta = Column(Float)
    ft_pct = Column(Float)
    efg_pct = Column(Float)
    orb = Column(Float)
    drb = Column(Float)
    reb = Column(Float)
    ast = Column(Float)
    stl = Column(Float)
    blk = Column(Float)
    tov = Column(Float)
    pf = Column(Float)
    pts = Column(Float)

    # Advanced stats (NBA_{season}_advanced.html)
    mp_total = Column(Float)   # total minutes in the season
    per = Column(Float)        # player efficiency rating
    ts_pct = Column(Float)     # true shooting %
    fg3a_rate = Column(Float)  # share of shots from three
    fta_rate = Column(Float)
    orb_pct = Column(Float)
    drb_pct = Column(Float)
    trb_pct = Column(Float)
    ast_pct = Column(Float)
    stl_pct = Column(Float)
    blk_pct = Column(Float)
    tov_pct = Column(Float)
    usg_pct = Column(Float)    # usage %
    ows = Column(Float)
    dws = Column(Float)
    ws = Column(Float)         # win shares
    ws_per_48 = Column(Float)
    obpm = Column(Float)
    dbpm = Column(Float)
    bpm = Column(Float)        # box plus/minus
    vorp = Column(Float)


class TeamSeason(Base):
    """Final standings: one row per team per season."""
    __tablename__ = "team_seasons"
    __table_args__ = (UniqueConstraint("team_abr", "season"), Index("idx_ts_season", "season"))

    id = Column(Integer, primary_key=True)
    team_abr = Column(Text, nullable=False)
    season = Column(Integer, nullable=False)
    franchise_id = Column(Text, nullable=False)  # current abbreviation, e.g. SEA -> OKC
    full_name = Column(Text)
    conference = Column(Text)                    # 'E' or 'W'
    division = Column(Text)
    w = Column(Integer)
    l = Column(Integer)  # noqa: E741  (losses; the standard basketball abbreviation)
    w_pct = Column(Float)
    gb = Column(Float)                           # games behind the division leader
    ps_g = Column(Float)                         # points scored per game
    pa_g = Column(Float)                         # points allowed per game
    srs = Column(Float)                          # simple rating system
    made_playoffs = Column(Boolean)


class Game(Base):
    """One row per played game (regular season, play-in, playoffs, Cup final)."""
    __tablename__ = "games"
    __table_args__ = (Index("idx_games_season", "season"), Index("idx_games_date", "game_date"))

    game_id = Column(Text, primary_key=True)  # BRef box-score id, e.g. 202310240DEN
    season = Column(Integer, nullable=False)
    game_date = Column(Date, nullable=False)
    away_team = Column(Text, nullable=False)
    home_team = Column(Text, nullable=False)
    away_name = Column(Text)
    home_name = Column(Text)
    away_score = Column(Integer)
    home_score = Column(Integer)
    home_win = Column(Boolean)
    overtimes = Column(Text)                  # e.g. "OT", "2OT"
    attendance = Column(Integer)
    remarks = Column(Text)                    # BRef notes: "Play-In Game", "at London, England"
    neutral = Column(Boolean)                 # played abroad or in the 2020 bubble
    game_type = Column(Text)                  # 'regular' | 'play_in' | 'playoff' | 'cup_final'


class WinTotal(Base):
    """Preseason betting line on each team's wins (sportsoddshistory.com), 1994 onward."""
    __tablename__ = "win_totals"

    season = Column(Integer, primary_key=True)
    team_abr = Column(Text, primary_key=True)
    line = Column(Float, nullable=False)       # e.g. 58.5
    over_odds = Column(Integer)                # American odds, e.g. -110
    under_odds = Column(Integer)
    actual_wins = Column(Integer)              # as reported by the odds site (validated vs standings)


class PlayerSalary(Base):
    """One row per player on a team's payroll that season (BRef team page "Salaries" table).
    bref_id has no foreign key: payroll includes players who never played (waived, injured)."""
    __tablename__ = "player_salaries"

    season = Column(Integer, primary_key=True)
    team_abr = Column(Text, primary_key=True)
    bref_id = Column(Text, primary_key=True)
    name = Column(Text)
    salary = Column(Integer, nullable=False)   # US dollars


class SalaryCap(Base):
    __tablename__ = "salary_caps"

    season = Column(Integer, primary_key=True)
    cap = Column(Integer, nullable=False)      # US dollars


class EtlCheckpoint(Base):
    """Which (season, step) the pipeline has finished, so a run can resume."""
    __tablename__ = "etl_checkpoint"

    season = Column(Integer, primary_key=True)
    step = Column(Text, primary_key=True)
    rows = Column(Integer)
    completed_at = Column(DateTime, server_default=func.now())
