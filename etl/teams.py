"""Franchise mapping: historical BRef abbreviations -> current franchise abbreviation."""

RELOCATIONS = {
    "NJN": "BRK",   # New Jersey Nets -> Brooklyn (2013)
    "CHH": "CHO",   # Charlotte Hornets (1989-2002); NBA credits this history to Charlotte
    "CHA": "CHO",   # Charlotte Bobcats (2005-2014)
    "NOH": "NOP",   # New Orleans Hornets
    "NOK": "NOP",   # New Orleans/Oklahoma City Hornets (2006-07)
    "SEA": "OKC",   # Seattle SuperSonics
    "VAN": "MEM",   # Vancouver Grizzlies
    "KCK": "SAC",   # Kansas City Kings
    "SDC": "LAC",   # San Diego Clippers
    "WSB": "WAS",   # Washington Bullets
}


def franchise_of(team_abr: str) -> str:
    return RELOCATIONS.get(team_abr, team_abr)
