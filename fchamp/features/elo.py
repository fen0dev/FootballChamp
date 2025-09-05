from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Elo:
    """
    🚀 ENHANCED ELO System with advanced features
    Completamente backward compatible: se non usi i nuovi parametri, funziona identico a prima
    """
    start: float = 1500.0
    k: float = 20.0
    hfa: float = 60.0
    mov_factor: float = 0.0
    adaptive_k: bool = False
    min_games_for_stability: int = 10
    uncertainty_bonus: float = 1.5
    home_away_split: bool = False
    ratings: Dict[str, float] = field(default_factory=dict)
    games_played: Dict[str, int] = field(default_factory=dict)
    home_ratings: Dict[str, float] = field(default_factory=dict)
    away_ratings: Dict[str, float] = field(default_factory=dict)

    def get(self, team: str, is_home: Optional[bool] = None) -> float:
        if self.home_away_split and is_home is not None:
            if is_home:
                return self.home_ratings.get(team, self.start)
            else:
                return self.away_ratings.get(team, self.start)

        return self.ratings.get(team, self.start)

    def expected_home_win(self, home: str, away: str) -> float:
        if self.home_away_split:
            home_rating = self.get(home, is_home=True)
            away_rating = self.get(away, is_home=False)
        else:
            home_rating = self.get(home)
            away_rating = self.get(away)

        # HFA dinamico basato su esperienza
        dynamic_hfa = self.hfa
        if self.adaptive_k:
            home_games = self.games_played.get(home, 0)
            if home_games < self.min_games_for_stability:
                # riduce HFA per squadre inesperte (piu' incertezza)
                experience_factor = home_games / self.min_games_for_stability
                dynamic_hfa = self.hfa * (0.7 + 0.3 * experience_factor)

        diff = (home_rating + dynamic_hfa) - away_rating
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _mov_scale(self, hg: int, ag: int) -> float:
        if self.mov_factor <= 0:
            return 1.0

        margin = abs(hg - ag)

        if margin <= 1:
            return 1.0
        else:
            return 1.0 + self.mov_factor * np.log(1 + margin - 1)

    def _get_k_factor(self, team: str) -> float:
        if not self.adaptive_k:
            return self.k

        games = self.games_played.get(team, 0)
        if games < self.min_games_for_stability:
            # K piu' alto per squadre inesperte 
            return self.k * self.uncertainty_bonus
        else:
            return self.k

    def update(self, home: str, away: str, hg: int, ag: int):
        p_home = self.expected_home_win(home, away)
        outcome = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        scale = self._mov_scale(hg, ag)

        if self.adaptive_k:
            k_home = self._get_k_factor(home)
            k_away = self._get_k_factor(away)
            delta_home = k_home * scale * (outcome - p_home)
            delta_away = k_away * scale * (p_home - outcome)
        else:
            # comportamento normale
            delta = self.k * scale * (outcome - p_home)
            delta_home = delta
            delta_away = -delta

        if self.home_away_split:
            # updating rating casa per home team e rating trasferta per away team
            current_home_home = self.home_ratings.get(home, self.start)
            self.home_ratings[home] = current_home_home + delta_home

            current_away_away = self.away_ratings.get(away, self.start)
            self.away_ratings[away] = current_away_away + delta_away

            self.ratings[home] = (self.home_ratings[home] + self.away_ratings.get(home, self.start)) / 2
            self.ratings[away] = (self.home_ratings.get(away, self.start) + self.away_ratings[away]) / 2
        else:
            self.ratings[home] = self.get(home) + delta_home
            self.ratings[away] = self.get(away) + delta_away

        # track games played (per adaptive features)
        if self.adaptive_k:
            self.games_played[home] = self.games_played.get(home, 0) + 1
            self.games_played[away] = self.games_played.get(away, 0) + 1

    def get_team_info(self, team: str):
        info = {
            'rating': self.get(team),
            'games_played': self.games_played.get(team, 0) if self.adaptive_k else None
        }

        if self.home_away_split:
            info.update({
                'home_rating': self.home_ratings.get(team, self.start),
                'away_rating': self.away_ratings.get(team, self.start),
                'home_away_diff': self.home_ratings.get(team, self.start) - self.away_ratings.get(team, self.start)
            })

        return info

    def get_ratings_summary(self) -> Dict[str, Dict]:
        all_teams = set(self.ratings.keys())

        if self.home_away_split:
            all_teams.update(self.home_ratings.keys())
            all_teams.update(self.away_ratings.keys())

        summary = {}
        for team in all_teams:
            summary[team] = self.get_team_info(team)

        return summary

    def apply_season_regression(self, teams: Set[str], regression_factor: float = 0.25):
        if regression_factor <= 0:
            return

        logger.info(f"🚀 Applying season regression (factor={regression_factor}) to {len(teams)} teams")

        for team in teams:
            # regression rating principale
            if team in self.ratings:
                current = self.ratings[team]
                self.ratings[team] = self.start + (current - self.start) * (1 - regression_factor)

            # regression home/away ratings se abilitati
            if self.home_away_split:
                if team in self.home_ratings:
                    current_home = self.home_ratings[team]
                    self.home_ratings[team] = self.start + (current_home - self.start) * (1 - regression_factor)

                if team in self.away_ratings:
                    current_away = self.away_ratings[team]
                    self.away_ratings[team] = self.start + (current_away - self.start) * (1 - regression_factor)

        logger.info(f"Applied season regression to {len(teams)} teams")