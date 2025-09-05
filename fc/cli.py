import typer
from pathlib import Path
from fc.config import load_config, AppConfig
from fc.logging_config import setup_logging
from fchamp.pipelines.train import run_train
from fchamp.pipelines.predict import run_predict
from fchamp.evaluation.backtest import run_backtest
from fchamp.pipelines.calibrate import run_calibration
from fchamp.data.loader import fetch_understat_results, load_matches, fetch_understat_league_shots_scrape
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from fchamp.features.advanced_stats import (
    add_shots_and_corners_features,
    add_head_to_head_stats,
    add_xg_proxy_features
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML Path"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    setup_logging(log_level)
    ctx.obj = load_config(config)

@app.command()
def backtest(ctx: typer.Context):
    """Run model backtest evaluation"""
    cfg: AppConfig = ctx.obj
    report = run_backtest(cfg)
    typer.echo(report)

@app.command()
def train(ctx: typer.Context):
    """Train a new model"""
    cfg: AppConfig = ctx.obj
    model_id = run_train(cfg)
    typer.echo(f"Model saved: {model_id}")

@app.command()
def predict(
    ctx: typer.Context,
    fixtures: Path = typer.Option(..., "--fixtures", help="CSV with date, home_team, away_team, etc..."),
    model_id: str = typer.Option(None, "--model-id", help="Model ID (default: last)"),
    output: Path = typer.Option("artifacts/predictions.csv", "--output", "-o", help="Output CSV path"),
):
    """Generate predictions for fixtures"""
    cfg: AppConfig = ctx.obj
    df = run_predict(cfg, fixtures_path=fixtures, model_id=model_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Predictions saved in: {output}")

@app.command()
def calibrate(
    ctx: typer.Context,
    model_id: str = typer.Option(None, "--model-id", help="Model ID (default: latest)"),
    ece_target: float = typer.Option(0.08, "--ece-target", help="ECE target to prefer method"),
):
    """Recalibra il modello corrente e salva calibrator.joblib"""
    cfg: AppConfig = ctx.obj
    res = run_calibration(cfg, model_id=model_id, ece_target=ece_target)
    typer.echo(res)

@app.command(name="fetch-xg")
def fetch_xg(
    ctx: typer.Context,
    league: str = typer.Option("Serie A", "--league"),
    seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons, es: 2023,2024"),
    output: Path = typer.Option("artifacts/xg.csv", "--output", "-o", help="Output CSV path"),
):
    """Scarica xG da web (adapter) e salva CSV."""
    _seasons = [int(s.strip()) for s in seasons.split(',')] if seasons else None
    df = fetch_understat_results(league=league, seasons=_seasons)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Saved xG to {output} ({len(df)} rows)")

@app.command(name="fetch-xg-scrape")
def fetch_xg_scrape(
    ctx: typer.Context,
    league: str = typer.Option("Serie A", "--league"),
    seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons, es: 2023,2024"),
    output: Path = typer.Option("artifacts/xg.csv", "--output", "-o", help="Output CSV path"),
):
    """Scrape understat.com (matchesData) e salva xG match-level in CSV."""
    from fchamp.data.loader import fetch_understat_league_xg_scrape
    _seasons = [int(s.strip()) for s in seasons.split(',')] if seasons else None
    df = fetch_understat_league_xg_scrape(league=league, seasons=_seasons)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Saved xG (scrape) to {output} ({len(df)} rows)")

@app.command(name="fetch-shots-scrape")
def fetch_shots_scrape(
    ctx: typer.Context,
    league: str = typer.Option("Serie A", "--league", "-l"),
    seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons, f.x.: 2023,2024"),
    output: Path = typer.Option("artifacts/shots.csv", "--output", "-o", help="Output CSV path"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Stampa avanzamento scraping"),
    limit: int = typer.Option(0, "--limit", help="Limita numero match da scaricare (0 = tutti)")
):
    """Scrape understat.com (match pages) e salva tiri match-level in CSV."""
    _seasons = [int(s.strip()) for s in seasons.split(',')] if seasons else None
    df = fetch_understat_league_shots_scrape(league=league, seasons=_seasons, progress=progress, limit=(limit or None))
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Saved SHOTS (scrape) to {output} ({len(df)} rows)")

@app.command()
def models(ctx: typer.Context):
    """🚀 List all available models with details"""
    cfg: AppConfig = ctx.obj
    from fchamp.models.registry import ModelRegistry
    
    reg = ModelRegistry(cfg.artifacts_dir)
    artifacts_dir = Path(cfg.artifacts_dir)
    
    if not artifacts_dir.exists():
        typer.echo("No models directory found.")
        return
    
    models = []
    for model_dir in artifacts_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "meta.json").exists():
            try:
                meta = json.loads((model_dir / "meta.json").read_text())
                models.append({
                    'id': model_dir.name,
                    'league': meta.get('league', 'unknown'),
                    'teams': len(meta.get('teams', [])),
                    'features': len(meta.get('features', [])),
                    'alpha': meta.get('alpha', 'N/A'),
                    'log_loss': meta.get('log_loss', 'N/A'),
                    'created': model_dir.stat().st_mtime
                })
            except Exception:
                continue
    
    if not models:
        typer.echo("No models found.")
        return
    
    # Sort by creation time (newest first)
    models.sort(key=lambda x: x['created'], reverse=True)
    
    # Rich table
    table = Table(title="🚀 Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("League", style="green")
    table.add_column("Teams", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Age", style="dim")
    
    from datetime import datetime, timedelta
    now = datetime.now().timestamp()
    
    for model in models:
        age_seconds = now - model['created']
        if age_seconds < 3600:
            age = f"{int(age_seconds/60)}m"
        elif age_seconds < 86400:
            age = f"{int(age_seconds/3600)}h"
        else:
            age = f"{int(age_seconds/86400)}d"
        
        table.add_row(
            model['id'][:20] + "..." if len(model['id']) > 23 else model['id'],
            model['league'],
            str(model['teams']),
            str(model['features']),
            f"{model['alpha']:.3f}" if isinstance(model['alpha'], float) else str(model['alpha']),
            age
        )
    
    console.print(table)
    
    latest_id = reg.get_latest_id()
    if latest_id:
        rprint(f"\n[green]Latest model: {latest_id}[/green]")

@app.command()
def info(
    ctx: typer.Context,
    model_id: str = typer.Option(None, "--model-id", help="Model ID (default: latest)")
):
    """🚀 Show detailed model information"""
    cfg: AppConfig = ctx.obj
    from fchamp.models.registry import ModelRegistry
    
    reg = ModelRegistry(cfg.artifacts_dir)
    if not model_id:
        model_id = reg.get_latest_id()
    
    if not model_id:
        typer.echo("No model found.")
        return
    
    model_dir = reg.model_dir(model_id)
    meta_file = model_dir / "meta.json"
    
    if not meta_file.exists():
        typer.echo(f"Model {model_id} not found.")
        return
    
    try:
        meta = json.loads(meta_file.read_text())
        
        # Model info table
        info_table = Table(title=f"🚀 Model Info: {model_id}")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        # Basic info
        info_table.add_row("League", meta.get('league', 'unknown'))
        info_table.add_row("Teams", str(len(meta.get('teams', []))))
        info_table.add_row("Features", str(len(meta.get('features', []))))
        info_table.add_row("Alpha", f"{meta.get('alpha', 'N/A'):.4f}" if isinstance(meta.get('alpha'), float) else str(meta.get('alpha', 'N/A')))
        info_table.add_row("Dixon-Coles", "Yes" if meta.get('use_dixon_coles', False) else "No")
        info_table.add_row("DC Rho", f"{meta.get('dc_rho', 'N/A'):.4f}" if isinstance(meta.get('dc_rho'), float) else str(meta.get('dc_rho', 'N/A')))
        info_table.add_row("Max Goals", str(meta.get('max_goals', 'N/A')))
        
        # Calibration info
        cal_info = meta.get('calibration', {})
        if cal_info.get('calibrated', False):
            info_table.add_row("Calibration", f"Yes ({cal_info.get('method', 'unknown')})")
        else:
            info_table.add_row("Calibration", "No")
        
        # GBM info
        gbm_info = meta.get('gbm', {})
        if gbm_info.get('enabled', False):
            info_table.add_row("GBM Ensemble", f"Yes (weight: {gbm_info.get('blend_weight', 'N/A')})")
        else:
            info_table.add_row("GBM Ensemble", "No")
        
        console.print(info_table)
        
        # Features table
        features = meta.get('features', [])
        if features:
            features_table = Table(title="Features Used")
            features_table.add_column("Feature", style="green")
            features_table.add_column("Type", style="dim")
            
            for feature in features:
                if 'elo' in feature.lower():
                    ftype = "ELO"
                elif 'roll' in feature.lower():
                    ftype = "Rolling"
                elif 'ewm' in feature.lower():
                    ftype = "EWM"
                elif 'book' in feature.lower():
                    ftype = "Market"
                else:
                    ftype = "Other"
                
                features_table.add_row(feature, ftype)
            
            console.print(features_table)
        
        # Teams
        teams = meta.get('teams', [])
        if teams:
            rprint(f"\n[green]Teams ({len(teams)}):[/green] {', '.join(teams[:10])}")
            if len(teams) > 10:
                rprint(f"[dim]... and {len(teams) - 10} more[/dim]")
        
    except Exception as e:
        typer.echo(f"Error reading model info: {e}")

@app.command()
def compare(
    ctx: typer.Context,
    fixtures: Path = typer.Option(..., "--fixtures", help="CSV with fixtures to compare"),
    models: str = typer.Option(None, "--models", help="Comma-separated model IDs (default: all)"),
    output: Path = typer.Option("artifacts/comparison.csv", "--output", "-o", help="Output comparison CSV")
):
    """🚀 Compare predictions from multiple models"""
    cfg: AppConfig = ctx.obj
    from fchamp.models.registry import ModelRegistry
    
    reg = ModelRegistry(cfg.artifacts_dir)
    
    # Get model list
    if models:
        model_ids = [m.strip() for m in models.split(',')]
    else:
        # Get all available models
        artifacts_dir = Path(cfg.artifacts_dir)
        model_ids = []
        if artifacts_dir.exists():
            for model_dir in artifacts_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "meta.json").exists():
                    model_ids.append(model_dir.name)
        
        # Limit to 3 most recent
        model_ids = sorted(model_ids, key=lambda x: (artifacts_dir / x).stat().st_mtime, reverse=True)[:3]
    
    if not model_ids:
        typer.echo("No models found for comparison.")
        return
    
    typer.echo(f"Comparing {len(model_ids)} models...")
    
    # Generate predictions for each model
    comparison_data = []
    fixtures_df = pd.read_csv(fixtures)
    
    for i, model_id in enumerate(model_ids):
        try:
            typer.echo(f"Generating predictions with model {i+1}/{len(model_ids)}: {model_id[:20]}...")
            df = run_predict(cfg, fixtures_path=fixtures, model_id=model_id)
            
            # Add model info to predictions
            df['model_id'] = model_id
            df['model_rank'] = i + 1
            
            comparison_data.append(df)
            
        except Exception as e:
            typer.echo(f"Error with model {model_id}: {e}")
            continue
    
    if not comparison_data:
        typer.echo("No successful predictions generated.")
        return
    
    # Combine all predictions
    combined_df = pd.concat(comparison_data, ignore_index=True)
    
    # Save detailed comparison
    output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output, index=False)
    
    # Summary comparison
    summary_table = Table(title="🚀 Model Comparison Summary")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Avg P(Home)", justify="right")
    summary_table.add_column("Avg P(Draw)", justify="right") 
    summary_table.add_column("Avg P(Away)", justify="right")
    summary_table.add_column("Avg λ(Home)", justify="right")
    summary_table.add_column("Avg λ(Away)", justify="right")
    
    for model_id in model_ids:
        model_data = combined_df[combined_df['model_id'] == model_id]
        if len(model_data) > 0:
            summary_table.add_row(
                model_id[:15] + "..." if len(model_id) > 18 else model_id,
                f"{model_data['p_home'].mean():.3f}",
                f"{model_data['p_draw'].mean():.3f}",
                f"{model_data['p_away'].mean():.3f}",
                f"{model_data['lambda_home'].mean():.2f}",
                f"{model_data['lambda_away'].mean():.2f}"
            )
    
    console.print(summary_table)
    typer.echo(f"\nDetailed comparison saved to: {output}")

@app.command(name="analyze-preds")
def analyze(
    ctx: typer.Context,
    predictions: Path = typer.Option(..., "--predictions", help="CSV with predictions to analyze"),
    actual_results: Path = typer.Option(None, "--actual", help="CSV with actual results (optional)")
):
    """🚀 Analyze prediction quality and patterns"""
    cfg: AppConfig = ctx.obj
    
    try:
        pred_df = pd.read_csv(predictions)
        
        # Basic analysis
        analysis_table = Table(title="🚀 Prediction Analysis")
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="white")
        
        # Prediction distribution
        analysis_table.add_row("Total Predictions", str(len(pred_df)))
        analysis_table.add_row("Avg P(Home)", f"{pred_df['p_home'].mean():.3f}")
        analysis_table.add_row("Avg P(Draw)", f"{pred_df['p_draw'].mean():.3f}")
        analysis_table.add_row("Avg P(Away)", f"{pred_df['p_away'].mean():.3f}")
        
        # Confidence metrics
        if 'prediction_confidence' in pred_df.columns:
            analysis_table.add_row("Avg Confidence", f"{pred_df['prediction_confidence'].mean():.3f}")
            high_conf = (pred_df['prediction_confidence'] > 0.6).sum()
            analysis_table.add_row("High Confidence (>60%)", f"{high_conf} ({high_conf/len(pred_df)*100:.1f}%)")
        
        # Market comparison (if available)
        if 'book_p_home' in pred_df.columns:
            model_home_avg = pred_df['p_home'].mean()
            market_home_avg = pred_df['book_p_home'].mean()
            analysis_table.add_row("Model vs Market (Home)", f"{model_home_avg:.3f} vs {market_home_avg:.3f}")
        
        # Lambda analysis
        analysis_table.add_row("Avg λ(Home)", f"{pred_df['lambda_home'].mean():.2f}")
        analysis_table.add_row("Avg λ(Away)", f"{pred_df['lambda_away'].mean():.2f}")
        
        console.print(analysis_table)
        
        # If actual results provided, calculate accuracy
        if actual_results and actual_results.exists():
            try:
                actual_df = pd.read_csv(actual_results)
                
                # Simple accuracy calculation
                if all(col in actual_df.columns for col in ['home_team', 'away_team', 'home_goals', 'away_goals']):
                    # Match predictions with actuals
                    merged = pred_df.merge(
                        actual_df, 
                        on=['home_team', 'away_team'], 
                        how='inner',
                        suffixes=('_pred', '_actual')
                    )
                    
                    if len(merged) > 0:
                        # Calculate actual outcomes
                        def get_outcome(row):
                            if row['home_goals'] > row['away_goals']:
                                return 0  # Home win
                            elif row['home_goals'] < row['away_goals']:
                                return 2  # Away win
                            else:
                                return 1  # Draw
                        
                        merged['actual_outcome'] = merged.apply(get_outcome, axis=1)
                        merged['predicted_outcome'] = merged[['p_home', 'p_draw', 'p_away']].idxmax(axis=1).map({'p_home': 0, 'p_draw': 1, 'p_away': 2})
                        
                        accuracy = (merged['actual_outcome'] == merged['predicted_outcome']).mean()
                        
                        rprint(f"\n[green]Accuracy on {len(merged)} matched predictions: {accuracy:.1%}[/green]")
                        
                        # Outcome breakdown
                        outcome_table = Table(title="Outcome Accuracy")
                        outcome_table.add_column("Outcome", style="cyan")
                        outcome_table.add_column("Predicted", justify="right")
                        outcome_table.add_column("Actual", justify="right")
                        outcome_table.add_column("Accuracy", justify="right")
                        
                        for outcome, name in [(0, "Home Win"), (1, "Draw"), (2, "Away Win")]:
                            pred_count = (merged['predicted_outcome'] == outcome).sum()
                            actual_count = (merged['actual_outcome'] == outcome).sum()
                            if pred_count > 0:
                                outcome_acc = ((merged['predicted_outcome'] == outcome) & (merged['actual_outcome'] == outcome)).sum() / pred_count
                            else:
                                outcome_acc = 0
                            
                            outcome_table.add_row(
                                name,
                                str(pred_count),
                                str(actual_count),
                                f"{outcome_acc:.1%}"
                            )
                        
                        console.print(outcome_table)
                
            except Exception as e:
                typer.echo(f"Error analyzing actual results: {e}")
        
    except Exception as e:
        typer.echo(f"Error analyzing predictions: {e}")

@app.command(name="analyze-league")
def analyze(
    ctx: typer.Context,
    league: str = typer.Option("ita", help="League: ita/epl/dsl"),
    team: str = typer.Option(None, help="Specific team to analyze")
):
    """🔍 Analizza statistiche avanzate per squadre"""
    
    # Usa il config dal context o carica direttamente
    config_path = f"config_{league}.yaml"
    cfg = load_config(config_path)
    df = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)
    
    # Applica tutte le feature
    df = add_shots_and_corners_features(df)
    df = add_head_to_head_stats(df)
    df = add_xg_proxy_features(df)
    
    if team:
        # Analisi specifica squadra
        team_home = df[df['home_team'] == team]
        team_away = df[df['away_team'] == team]
        
        print(f"\n📊 Analisi per {team}")
        print(f"Partite in casa: {len(team_home)}")
        print(f"Partite fuori: {len(team_away)}")
        
        if 'home_xg_proxy' in df.columns:
            avg_xg_home = team_home['home_xg_proxy'].mean()
            avg_xg_away = team_away['away_xg_proxy'].mean()
            print(f"xG medio in casa: {avg_xg_home:.2f}")
            print(f"xG medio fuori: {avg_xg_away:.2f}")
        
        if 'home_shots_roll' in df.columns:
            recent_form = team_home['home_shots_roll'].iloc[-1]
            print(f"Media tiri recenti (casa): {recent_form:.1f}")
    else:
        # Analisi generale lega
        print(f"\n📊 Statistiche {league.upper()}")
        print(f"Totale partite: {len(df)}")
        print(f"Media gol per partita: {(df['FTHG'] + df['FTAG']).mean():.2f}")
        
        if 'home_xg_proxy' in df.columns:
            print(f"xG medio casa: {df['home_xg_proxy'].mean():.2f}")
            print(f"xG medio trasferta: {df['away_xg_proxy'].mean():.2f}")

def main():
    app()

if __name__ == '__main__':
    main()